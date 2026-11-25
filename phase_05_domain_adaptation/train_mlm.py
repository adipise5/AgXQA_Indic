"""Train the final XLM-R encoder layer on the reviewed local MLM pilot."""

import argparse
import json
import math
import platform
from pathlib import Path
from tempfile import TemporaryDirectory

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError
from tokenizers import Tokenizer

from prepare_mlm_data import sha256


MODEL_ID = 'FacebookAI/xlm-roberta-base'
REVISION = 'e73636d4f797dec63c3081bb6ed5c7b0bb3f2089'
WEIGHTS_SHA256 = '6fd4797bc397c3b8b55d6bb5740366b57e6a3ce91c04c77f22aafc0c128e6feb'


def read_splits(inputs, manifest):
    if manifest['status'] != 'local_mlm_input_prepared':
        raise ValueError('Expected a prepared MLM input manifest')
    splits = {}
    for split in ('train', 'validation'):
        source = inputs / f'{split}.jsonl'
        if sha256(source) != manifest['splits'][split]['sha256']:
            raise ValueError(f'MLM input checksum mismatch for {split}')
        rows = [json.loads(line) for line in source.read_text(encoding='utf-8').splitlines()]
        if not rows or len(rows) != manifest['splits'][split]['sequences'] or any(
            row['split'] != split for row in rows
        ):
            raise ValueError(f'Invalid MLM sequences for {split}')
        splits[split] = rows
    for key in ('document_id', 'document_family_id'):
        if {r[key] for r in splits['train']} & {r[key] for r in splits['validation']}:
            raise ValueError(f'MLM train and validation share {key}')
    return splits


def mask_rows(rows, torch, seed, mask_id, replacement_ids):
    generator = torch.Generator().manual_seed(seed)
    batches = []
    for row in rows:
        ids = torch.tensor(row['input_ids'], dtype=torch.long)
        attention = torch.tensor(row['attention_mask'], dtype=torch.long)
        special = torch.tensor(row['special_tokens_mask'], dtype=torch.bool)
        eligible = (attention.bool() & ~special).nonzero(as_tuple=True)[0]
        if not len(eligible):
            raise ValueError(f"No maskable tokens for {row['document_id']}")
        positions = eligible[torch.rand(len(eligible), generator=generator) < 0.15]
        if not len(positions):
            positions = eligible[torch.randint(len(eligible), (1,), generator=generator)]
        labels = ids[positions].clone()
        choices = torch.rand(len(positions), generator=generator)
        ids[positions[choices < 0.8]] = mask_id
        random_positions = positions[(choices >= 0.8) & (choices < 0.9)]
        ids[random_positions] = replacement_ids[torch.randint(len(replacement_ids), (len(random_positions),),
                                                               generator=generator)]
        batches.append((ids.unsqueeze(0), attention.unsqueeze(0), positions, labels))
    return batches


def loss_for_batch(model, batch, device, torch):
    ids, attention, positions, labels = (tensor.to(device) for tensor in batch)
    encoded = model.roberta(input_ids=ids, attention_mask=attention).last_hidden_state
    logits = model.lm_head(encoded[0, positions])
    loss = torch.nn.functional.cross_entropy(logits.float(), labels, reduction='sum')
    if not torch.isfinite(loss).item():
        raise ValueError('Non-finite MLM loss')
    return loss, labels.numel()


def mean_loss(model, batches, device, torch, optimizer=None, parameters=()):
    # The frozen layers use evaluation mode throughout. Only the trained layer uses dropout.
    model.eval()
    model.roberta.encoder.layer[-1].train(optimizer is not None)
    total_loss = 0.0
    total_labels = 0
    with torch.set_grad_enabled(optimizer is not None):
        for batch in batches:
            loss, count = loss_for_batch(model, batch, device, torch)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                (loss / count).backward()
                torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
                optimizer.step()
            total_loss += loss.detach().item()
            total_labels += count
    return {'loss': total_loss / total_labels, 'masked_tokens': total_labels}


def train(model, splits, destination, device, seed, epochs, learning_rate, mask_id, replacement_ids, provenance):
    import torch
    from safetensors.torch import load_file, save_file

    if destination.exists():
        raise ValueError(f'Output already exists: {destination}')
    torch.manual_seed(seed)
    model.requires_grad_(False)
    layer = model.roberta.encoder.layer[-1].requires_grad_(True)
    parameter_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    initial = {name: tensor.detach().cpu().clone() for name, tensor in layer.state_dict().items()}
    model.to(device)
    parameters = list(layer.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate)
    validation = mask_rows(splits['validation'], torch, seed + 1, mask_id, replacement_ids)
    baseline = mean_loss(model, validation, device, torch)
    order = torch.Generator().manual_seed(seed)
    history = []
    for epoch in range(epochs):
        indices = torch.randperm(len(splits['train']), generator=order).tolist()
        rows = [splits['train'][i] for i in indices]
        batches = mask_rows(rows, torch, seed + 2 + epoch, mask_id, replacement_ids)
        training = mean_loss(model, batches, device, torch, optimizer, parameters)
        scores = mean_loss(model, validation, device, torch)
        history.append({'epoch': epoch + 1, 'train': training, 'validation': scores})
        print(json.dumps(history[-1]), flush=True)

    adapted = {name: tensor.detach().cpu().contiguous() for name, tensor in layer.state_dict().items()}
    if all(torch.equal(value, initial[name]) for name, value in adapted.items()):
        raise ValueError('MLM optimization did not change the selected layer')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination.parent) as staging:
        output = Path(staging) / 'run'
        output.mkdir()
        checkpoint = output / 'final_encoder_layer.safetensors'
        save_file(adapted, str(checkpoint))
        layer.load_state_dict(initial, strict=True)
        layer.load_state_dict(load_file(str(checkpoint)), strict=True)
        restored = mean_loss(model, validation, device, torch)
        if not math.isclose(restored['loss'], history[-1]['validation']['loss'], rel_tol=1e-5, abs_tol=1e-6):
            raise ValueError('Reloaded checkpoint did not reproduce the validation loss')
        result = {
            **provenance, 'device': device, 'platform': platform.platform(), 'torch': torch.__version__,
            'seed': seed, 'validation_mask_seed': seed + 1, 'epochs': epochs,
            'optimizer': 'AdamW', 'learning_rate': learning_rate, 'batch_size': 1,
            'mask_probability': 0.15, 'trained_parameters': sum(p.numel() for p in parameters),
            'parameter_names': parameter_names, 'baseline_validation': baseline, 'history': history,
            'reloaded_validation': restored, 'checkpoint_sha256': sha256(checkpoint),
        }
        (output / 'metrics.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        output.rename(destination)
    return result


def load_base(model_dir, cache_dir):
    from transformers import XLMRobertaForMaskedLM

    if model_dir is None:
        try:
            for name in ('config.json', 'model.safetensors'):
                artifact = hf_hub_download(MODEL_ID, name, revision=REVISION, cache_dir=cache_dir)
        except (HfHubHTTPError, LocalEntryNotFoundError, OSError) as exc:
            raise ValueError(f'Pinned XLM-R snapshot unavailable. Supply a local snapshot via --model-dir. {exc}') from exc
        model_dir = Path(artifact).parent
    weights = model_dir / 'model.safetensors'
    if sha256(weights) != WEIGHTS_SHA256:
        raise ValueError('Local model weights do not match the pinned XLM-R checkpoint')
    model, info = XLMRobertaForMaskedLM.from_pretrained(
        str(model_dir), local_files_only=True, use_safetensors=True, output_loading_info=True,
    )
    if info['missing_keys'] or info['mismatched_keys'] or info['error_msgs']:
        raise ValueError(f'Incomplete pretrained model load: {info}')
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('phase_05_domain_adaptation/config.json'))
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--model-dir', type=Path)
    parser.add_argument('--cache-dir', type=Path)
    parser.add_argument('--device', choices=('cpu', 'cuda', 'mps'), default='cpu')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--learning-rate', type=float, default=2e-5)
    args = parser.parse_args()
    try:
        import torch
        import transformers

        cfg = json.loads(args.config.read_text(encoding='utf-8'))
        manifest_path = args.inputs / 'manifest.json'
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if args.epochs < 1 or not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
            raise ValueError('Epochs and learning rate must be positive')
        spec = manifest['tokenizer']
        if (spec['id'], spec['revision']) != (MODEL_ID, REVISION):
            raise ValueError('MLM tokenizer and XLM-R base revision must match')
        splits = read_splits(args.inputs, manifest)
        tokenizer_path = Path(hf_hub_download(MODEL_ID, 'tokenizer.json', revision=REVISION, cache_dir=args.cache_dir))
        if sha256(tokenizer_path) != spec['sha256']:
            raise ValueError('Tokenizer checksum does not match the prepared inputs')
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        mask_id = tokenizer.token_to_id('<mask>')
        special = {tokenizer.token_to_id(token) for token in ('<s>', '</s>', '<pad>', '<unk>', '<mask>')}
        replacement_ids = torch.tensor(sorted(set(tokenizer.get_vocab().values()) - special), dtype=torch.long)
        destination = Path(cfg['output_dir']) / 'mlm_runs' / (
            f'{args.inputs.resolve().name}-last-layer-{args.epochs}ep-lr{args.learning_rate:g}-seed{cfg["seed"]}-{args.device}'
        )
        model = load_base(args.model_dir, args.cache_dir)
        result = train(model, splits, destination, args.device, cfg['seed'], args.epochs, args.learning_rate,
                       mask_id, replacement_ids, {
                           'status': 'local_pretrained_mlm_pilot', 'model_id': MODEL_ID, 'model_revision': REVISION,
                           'model_sha256': WEIGHTS_SHA256, 'input_manifest_sha256': sha256(manifest_path),
                           'phase4_version': manifest['phase4_version'], 'phase6_baseline': manifest['phase6_baseline'],
                           'transformers': transformers.__version__,
                       })
    except (ImportError, ValueError, OSError, KeyError, HfHubHTTPError) as exc:
        parser.exit(2, f'{exc}\n')
    print(json.dumps({'output_dir': str(destination), **result}, indent=2))


if __name__ == '__main__':
    main()
