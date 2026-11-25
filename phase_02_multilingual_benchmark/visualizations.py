"""Plots use only observed counts/judgments; absent measurements become notices."""
import os
from pathlib import Path


def generate_plots(report,directory):
    directory = Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR",str(directory.parent.parent/".cache/matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME",str(directory.parent.parent/".cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size":11,"figure.dpi":150})

    def notice(name,title,message):
        fig,ax = plt.subplots(figsize=(8,4.5))
        ax.axis("off")
        ax.text(.5,.67,title,ha="center",fontsize=17,weight="bold",transform=ax.transAxes)
        ax.text(.5,.37,message,ha="center",va="center",linespacing=1.6,transform=ax.transAxes)
        fig.tight_layout();fig.savefig(directory/(name+".png"));plt.close(fig)

    languages = ["en","hi","mr","bn","ta"]
    counts = [sum(v["candidate_rows"] for k,v in report["observed"].items() if k.startswith(language+"/")) for language in languages]
    fig,ax = plt.subplots(figsize=(8,4.5))
    bars=ax.bar(languages,counts,color=["#276f9b","#bf6b35","#54856b","#9679ad","#b18b3d"])
    ax.bar_label(bars,padding=3)
    ax.set(title="Observed prepared source / machine-candidate rows",ylabel="Rows",ylim=(0,max(2300,max(counts)*1.15)))
    fig.text(.5,.02,"These are observed preparation counts, not a claim of final human-approved benchmark coverage.",ha="center",fontsize=8)
    fig.tight_layout(rect=(0,.05,1,1));fig.savefig(directory/"language_distribution.png");plt.close(fig)
    reviewed = [language for language in languages[1:] if report["human_quality"][language]["review_judgments"]]
    if not reviewed:
        notice("translation_quality","No human quality ratings yet","Semantic equivalence, fluency and terminology scores\nwill be plotted only after actual reviewer judgments exist.")
        notice("human_review_outcomes","No completed human decisions yet","Accept / Correct / Reject outcomes have not been measured.\nUnfilled worksheets are not completed reviews.")
    else:
        fig,ax=plt.subplots(figsize=(8,4.5))
        for field in ("semantic_equivalence","fluency","terminology"):
            ax.plot(reviewed,[report["human_quality"][l][field] for l in reviewed],marker="o",label=field.replace("_"," "))
        ax.set(ylim=(1,5),ylabel="Mean actual human rating (1–5)",title="Human translation-quality judgments")
        ax.legend();fig.tight_layout();fig.savefig(directory/"translation_quality.png");plt.close(fig)
        fig,ax=plt.subplots(figsize=(8,4.5))
        for field in ("acceptance_without_correction_rate","correction_rate","rejection_rate"):
            ax.plot(reviewed,[100*report["human_quality"][l][field] for l in reviewed],marker="o",label=field.replace("_"," "))
        ax.set(ylim=(0,100),ylabel="Percentage of actual review judgments",title="Human decisions")
        ax.legend(fontsize=8);fig.tight_layout();fig.savefig(directory/"human_review_outcomes.png");plt.close(fig)
    translated = {language:[v for k,v in report["observed"].items() if k.startswith(language+"/") and v["candidate_rows"]] for language in languages[1:]}
    if not any(translated.values()):
        notice("answer_span_validity","No translated answer spans yet","Machine translation has not been run.\nBefore/after correction validity is not measured.")
    elif any(v["accepted_review_resolved_pairs"] for v in report["paired_span_validity"].values()):
        paired=report["paired_span_validity"]
        available=[l for l in languages[1:] if paired[l]["accepted_review_resolved_pairs"]]
        fig,ax=plt.subplots(figsize=(8,4.5))
        positions=list(range(len(available)))
        for delta,field,label,color in [(-.18,"before_valid_rows","Before correction","#bf6b35"),(.18,"after_valid_rows","After correction","#276f9b")]:
            rates=[100*paired[l][field]/paired[l]["accepted_review_resolved_pairs"] for l in available]
            bars=ax.bar([p+delta for p in positions],rates,width=.36,label=label,color=color)
            ax.bar_label(bars,fmt="%.1f",padding=3)
        ax.set(xticks=positions,xticklabels=available,ylim=(0,112),ylabel="Valid-span records (%)",title="Same review-resolved records before / after correction")
        ax.legend(fontsize=8)
        fig.text(.5,.02,"Accepted review-resolved pairs only; rejects/unresolved records excluded. See report for counts and full coverage.",ha="center",fontsize=7)
        fig.tight_layout(rect=(0,.05,1,1));fig.savefig(directory/"answer_span_validity.png");plt.close(fig)
    else:
        available=[l for l in languages[1:] if translated[l]]
        rates=[100*sum(v["valid_span_rows"] for v in translated[l])/sum(v["candidate_rows"] for v in translated[l]) for l in available]
        fig,ax=plt.subplots(figsize=(8,4.5));bars=ax.bar(available,rates,color="#276f9b")
        ax.bar_label(bars,fmt="%.1f",padding=3)
        ax.set(ylim=(0,105),ylabel="Valid-span rows / observed candidate rows (%)",title="Machine-candidate span validity; human after-correction curve unavailable")
        fig.tight_layout();fig.savefig(directory/"answer_span_validity.png");plt.close(fig)
    errors=report["human_error_counts"] or report["error_counts"]
    if errors:
        names=list(errors);fig,ax=plt.subplots(figsize=(9,max(4,len(names)*.3)))
        ax.barh(names,[errors[n] for n in names],color="#bf6b35")
        ax.set(xlabel="Observed judgments / flagged records",title="Human error categories" if report["human_error_counts"] else "Automated issue flags (not human error ratings)")
        fig.tight_layout();fig.savefig(directory/"error_distribution.png");plt.close(fig)
    else:
        notice("error_distribution","No observed issue counts yet","Absence of translated records is not evidence of error-free translations.")
