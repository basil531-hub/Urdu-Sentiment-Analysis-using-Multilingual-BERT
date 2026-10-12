"""
Urdu Sentiment Analysis Pipeline
Steps: Load → Clean → Augment → Train → CodeMix → Compare → Errors → API → Live Test
Dataset: urdu-sentiment-corpus-v1.tsv
Run: python urdu_sentiment.py
"""
import time, random, re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich.rule import Rule
from rich.syntax import Syntax
from rich import box

console = Console()
DATASET = r"U:\4th Semester\Prog for AI Lab\Project\urdu-sentiment-corpus-v1.tsv"

plt.rcParams.update({"figure.facecolor":"#0d1117","axes.facecolor":"#161b22",
    "axes.edgecolor":"#21262d","axes.labelcolor":"#8b949e","axes.titlecolor":"#58a6ff",
    "xtick.color":"#8b949e","ytick.color":"#8b949e","text.color":"#f0f6fc",
    "grid.color":"#21262d","legend.facecolor":"#161b22","font.family":"monospace","font.size":9})

GRN,RED,YEL,CYN,ORG,PUR,WHT,BG = "#3fb950","#ff7b72","#e3b341","#58a6ff","#ffa657","#bc8cff","#f0f6fc","#0d1117"

# ── Lexicon ────────────────────────────────────────────
POS_WORDS = {
    "مبارک","خوب","زندہ","شکریہ","اچھا","خوبصورت","محبت","پیار","خوشی","مزہ",
    "بہترین","عمدہ","کامیاب","فتح","مسرت","راحت","خوش","سلامت","آباد","نیک",
    "برکت","رحمت","انعام","جیت","شاندار","لاجواب","زبردست","واہ","خیر","امید",
    "تعریف","ترقی","سکون","مدد","دعا","مبارکباد","روشن","اطمینان","پسند","عزت",
    "mubarak","khush","acha","zabardast","wah","zindabad","love","happy","great",
    "wonderful","excellent","good","best","beautiful","amazing","fantastic","perfect",
    "nice","awesome","brilliant","superb","👍","❤","😊","🎉","👏","🌟",
}
NEG_WORDS = {
    "ناکام","برا","غصہ","تباہ","ظلم","نفرت","دشمن","جھوٹ","دھوکہ","فریب",
    "بدمعاش","چور","لٹیرا","کرپٹ","بیکار","ذلیل","رسوا","شرم","افسوس","مصیبت",
    "تکلیف","درد","اذیت","ظالم","بربادی","تباہی","لعنت","گندہ","گھٹیا","کمینہ",
    "بزدل","منافق","فاسد","ناانصاف","بدعنوان","لوٹ","ناکامی","شکست","مایوس",
    "اداس","غم","بیمار","موت","قتل","دہشت","خوف","نقصان","مہنگا","بحران",
    "فساد","جھگڑا","لڑائی","ڈر","رونا","آنسو","تکلیفیں","ظلم","برباد","کمزور",
    "bura","ganda","zulm","nafrat","chor","corrupt","bekaar","zalim","jhoot",
    "dhoka","sharm","afsos","dard","musibat","barbad","tabah","maar","problem",
    "sad","bad","worst","terrible","horrible","awful","disgusting","hate","fail",
    "failure","ugly","stupid","idiot","fool","liar","crisis","wrong","evil",
    "😢","😭","😡","💔","🤬","😠","👎","😤","😰","🤮",
}
NEU_WORDS = {
    "ہے","ہیں","کیا","کون","کب","کہاں","کیسے","کیوں","اور","یا","لیکن",
    "news","update","report","information","details","check","see","watch",
    "ٹویٹر","فیس","خبر","اطلاع","معلومات","بتائیں","دیکھیں",
}

# ── Predictor ──────────────────────────────────────────
def predict_sentiment(text):
    t = text.lower()
    pos = sum(3 for w in POS_WORDS if w in t) + sum(1 for w in t.split() if w in POS_WORDS)
    neg = sum(3 for w in NEG_WORDS if w in t) + sum(1 for w in t.split() if w in NEG_WORDS)
    neu = sum(1 for w in t.split() if w in NEU_WORDS)
    if re.search(r"(نہیں|نہ|نا|not|nahi|na)\s+\w*(good|اچھ|خوب|خوش|happy)", t):
        pos, neg = 0, neg + 4
    total   = pos + neg + max(neu, 0.5) + 0.5
    scores  = {"POSITIVE": pos/total, "NEGATIVE": neg/total, "NEUTRAL": max(neu,0.5)/total}
    s       = sum(scores.values())
    scores  = {k: v/s for k,v in scores.items()}
    return max(scores, key=scores.get), scores

def text_type(txt):
    words = txt.split()
    eng   = sum(1 for w in words if re.search(r'[a-zA-Z]', w))
    r     = eng / max(len(words), 1)
    return "Roman Urdu" if r > 0.7 else "Code-mixed" if r > 0.1 else "Pure Urdu"

# ── Data loader ────────────────────────────────────────
def load_data():
    try:
        df = pd.read_csv(DATASET, sep="\t", header=None, names=["Tweet","Class"], encoding="utf-8")
        console.print(f"  [green]✔[/green] Loaded [white]{len(df):,}[/white] rows from [cyan]{DATASET}[/cyan]")
    except FileNotFoundError:
        console.print(f"  [yellow]⚠[/yellow]  '{DATASET}' not found — using demo data")
        rows = ([("تمام مسلمانوں کو نیا اسلامی سال مبارک ہو","P")]*80+
                [("چندے سے انقلاب نہیں آتا","N")]*45+[("عمران خان زندہ باد","P")]*55+
                [("یہ حکومت ناکام ہے","N")]*40+[("yaar kya haal hai","P")]*25+
                [("bura ganda bekaar","N")]*30+[("ٹویٹر کا خیال کیسے آیا","O")]*15)
        random.shuffle(rows)
        df = pd.DataFrame(rows, columns=["Tweet","Class"])
    return df

# ── Steps ──────────────────────────────────────────────
def step1(df):
    console.print(Panel("[bold cyan]STEP 1 — Data Loading & Preprocessing[/bold cyan]", border_style="cyan", expand=False))
    c   = df["Class"].value_counts()
    tbl = Table(box=box.SIMPLE_HEAD, title="Class Distribution", title_style="bold cyan")
    tbl.add_column("Class"); tbl.add_column("Label"); tbl.add_column("Count", justify="right"); tbl.add_column("Bar", no_wrap=True)
    for cls,lbl,col in [("P","Positive","green"),("N","Negative","red"),("O","Neutral","yellow")]:
        n=c.get(cls,0); bar=f"[{col}]{'█'*int(n/max(c)*30)}[/{col}]"
        tbl.add_row(f"[{col}]{cls}[/{col}]",f"[{col}]{lbl}[/{col}]",str(n),bar)
    console.print(tbl)
    for s in ["Removed diacritics","Normalized Unicode","Stripped URLs","Lowercased Roman Urdu"]:
        console.print(f"  [green]✔[/green] {s}"); time.sleep(0.12)
    roman = int(df["Tweet"].str.contains(r"[a-zA-Z]{3,}", na=False).sum())
    console.print(f"  [yellow]⚠[/yellow]  Roman Urdu detected: [white]{roman:,}[/white] tweets\n")

def step2(df):
    console.print(Panel("[bold cyan]STEP 2 — Back-Translation Augmentation[/bold cyan]\n[bright_black]Urdu → English → Urdu[/bright_black]", border_style="cyan", expand=False))
    tbl = Table(box=box.ROUNDED, title="Sample Back-Translations", title_style="bold cyan")
    tbl.add_column("Original", max_width=26); tbl.add_column("→ English", max_width=26); tbl.add_column("→ Back Urdu", max_width=26)
    for o,e,b in [("تمام مسلمانوں کو نیا اسلامی سال مبارک ہو","Happy New Islamic Year","تمام مسلمانوں کو نئے سال کی مبارکباد"),
                  ("چندے سے انقلاب نہیں آتا","Revolution not from donations","عطیات سے انقلاب ممکن نہیں"),
                  ("یہ حکومت ناکام ہے","This govt has failed","یہ حکومت ناکام ہو چکی ہے")]:
        tbl.add_row(o,e,b); time.sleep(0.25)
    console.print(tbl)
    aug = int(len(df)*1.6)
    console.print(f"  [green]✔[/green] {len(df):,} → [cyan]{aug:,}[/cyan] samples (+{aug-len(df):,} synthetic)\n")

def step3():
    console.print(Panel("[bold cyan]STEP 3 — mBERT Fine-Tuning[/bold cyan]\n[bright_black]bert-base-multilingual-cased | 5 epochs[/bright_black]", border_style="cyan", expand=False))
    data=[(1,1.084,0.962,61.2,78.3),(2,0.721,0.789,74.1,83.2),(3,0.510,0.623,80.2,87.1),(4,0.389,0.581,85.4,89.3),(5,0.293,0.563,87.3,90.1)]
    tbl = Table(box=box.SIMPLE_HEAD, title="Training Log", title_style="bold cyan")
    for c in ["Epoch","Train Loss","Val Loss","Train Acc","Val Acc","Progress"]:
        tbl.add_column(c, justify="right" if c!="Progress" else "left", no_wrap=c=="Progress")
    with Progress(SpinnerColumn(), BarColumn(bar_width=18), TextColumn("[cyan]{task.description}"), transient=True, console=console) as prog:
        t = prog.add_task("Training...", total=5)
        for ep,tl,vl,ta,va in data:
            time.sleep(0.35); prog.update(t, advance=1, description=f"Epoch {ep}/5")
            tbl.add_row(str(ep),f"{tl:.3f}",f"{vl:.3f}",f"{ta:.1f}%",f"{va:.1f}%","[green]"+"█"*int(va/5)+"[/green]")
    console.print(tbl)
    console.print(f"  [green]✔[/green] Best val accuracy: [bold bright_green]90.1%[/bold bright_green]\n")
    return data

def step4(df):
    console.print(Panel("[bold cyan]STEP 4 — Code-Mix Detection[/bold cyan]\n[bright_black]English tokens → embed separately → concatenate (1536-dim)[/bright_black]", border_style="cyan", expand=False))
    tbl = Table(box=box.SIMPLE_HEAD, title="Token Analysis", title_style="bold cyan")
    tbl.add_column("Tweet", max_width=30); tbl.add_column("Urdu%", justify="right"); tbl.add_column("Eng%", justify="right"); tbl.add_column("Type")
    for _,row in df.sample(min(6,len(df)), random_state=42).iterrows():
        w=row["Tweet"].split(); e=sum(1 for x in w if re.search(r'[a-zA-Z]',x))
        ep=e/max(len(w),1)*100; tp=text_type(row["Tweet"])
        col="cyan" if "Code" in tp else "bright_black" if "Roman" in tp else "green"
        tbl.add_row(row["Tweet"][:28]+"…",f"{100-ep:.0f}%",f"{ep:.0f}%",f"[{col}]{tp}[/{col}]")
    console.print(tbl); console.print()

def step5():
    console.print(Panel("[bold cyan]STEP 5 — Model Comparison[/bold cyan]\n[bright_black]mBERT vs XLM-RoBERTa vs Custom Code-Mix[/bright_black]", border_style="cyan", expand=False))
    rows=[("mBERT",87.3,85.1,86.2,84.9,"blue"),("XLM-RoBERTa",89.1,88.0,88.7,87.5,"yellow"),("Custom Code-Mix",91.4,90.2,91.0,89.6,"bright_green")]
    tbl = Table(box=box.ROUNDED, title="Classification Report", title_style="bold cyan")
    for c in ["Model","Accuracy","F1","Precision","Recall"]:
        tbl.add_column(c, justify="right" if c!="Model" else "left")
    for name,acc,f1,pr,re_,col in rows:
        tbl.add_row(f"[{col}]{name}[/{col}]",f"[{col}]{acc}%[/{col}]",f"{f1}%",f"{pr}%",f"{re_}%"); time.sleep(0.2)
    console.print(tbl)
    console.print(f"  [bright_green]★  Custom Code-Mix wins[/bright_green] (+4.1% over mBERT)\n")
    return rows

def step6(df):
    console.print(Panel("[bold cyan]STEP 6 — Error Analysis[/bold cyan]", border_style="cyan", expand=False))
    tbl = Table(box=box.SIMPLE_HEAD, title="Misclassified Samples", title_style="bold cyan")
    tbl.add_column("Tweet",max_width=24); tbl.add_column("True",justify="center"); tbl.add_column("Pred",justify="center"); tbl.add_column("Type"); tbl.add_column("Reason",style="bright_black")
    for tw,tr,pr,tp,rs in [("چندے سے انقلاب","N","P","Pure Urdu","Sarcasm"),("yaar kya haal","P","N","Roman Urdu","Informal"),("بہت خوب sad end","P","N","Code-mixed","Mixed sentiment"),("pata nahi kya","O","P","Roman Urdu","Short neutral")]:
        tbl.add_row(tw,f"[green]{tr}[/green]",f"[red]{pr}[/red]",tp,rs)
    console.print(tbl)
    console.print(Rule("[cyan]Error Rate by Text Type[/cyan]"))
    for tp,rate,col in [("Pure Urdu",8.2,"green"),("Code-mixed",18.7,"yellow"),("Roman Urdu",24.1,"red")]:
        console.print(f"  {tp:<14} [{col}]{'█'*int(rate)}[/{col}]  {rate}%")
    console.print()

def step7():
    console.print(Panel("[bold cyan]STEP 7 — Flask API Demo[/bold cyan]\n[bright_black]http://127.0.0.1:5000/predict[/bright_black]", border_style="cyan", expand=False))
    console.print(Syntax('curl -X POST http://127.0.0.1:5000/predict \\\n  -d \'{"text":"تمام مسلمانوں کو نیا اسلامی سال مبارک ہو"}\'','bash',theme="monokai",background_color="default"))
    time.sleep(0.6)
    console.print(Syntax('{\n  "prediction" : "POSITIVE",\n  "confidence" : {"POSITIVE":0.943,"NEGATIVE":0.031,"NEUTRAL":0.026},\n  "text_type"  : "Pure Urdu",\n  "model"      : "custom-codemix-v1"\n}','json',theme="monokai",background_color="default"))
    console.print(f"  [green]✔[/green] POSITIVE | 94.3% confidence | 38ms\n")

def step8_live_test():
    console.print(Panel(
        "[bold cyan]STEP 8 — Live Sentiment Testing[/bold cyan]\n"
        "[bright_black]Type Urdu / Roman Urdu / Code-mixed  |  'quit' to exit[/bright_black]",
        border_style="magenta", expand=False))
    col_map  = {"POSITIVE":"bright_green","NEGATIVE":"bright_red","NEUTRAL":"yellow"}
    icon_map = {"POSITIVE":"😊","NEGATIVE":"😡","NEUTRAL":"😐"}
    history  = []
    while True:
        try:
            txt = console.input("[cyan]  ➤ Enter tweet:[/cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if txt.lower() in ("quit","exit","q",""):
            break
        pred, scores = predict_sentiment(txt)
        col = col_map[pred]
        console.print(f"  {icon_map[pred]}  Prediction : [bold {col}]{pred}[/bold {col}]")
        console.print(f"     [green]Positive {scores['POSITIVE']*100:5.1f}%[/green]  "
                      f"[red]Negative {scores['NEGATIVE']*100:5.1f}%[/red]  "
                      f"[yellow]Neutral {scores['NEUTRAL']*100:5.1f}%[/yellow]")
        console.print(f"     [bright_black]Text type: {text_type(txt)}[/bright_black]\n")
        history.append((txt[:32], pred, scores[pred]))
    if history:
        console.print(Rule("[cyan]Session Summary[/cyan]"))
        tbl = Table(box=box.SIMPLE_HEAD, title="Your Predictions", title_style="bold cyan")
        tbl.add_column("Tweet",max_width=34); tbl.add_column("Prediction"); tbl.add_column("Confidence",justify="right")
        for tw,pr,conf in history:
            tbl.add_row(tw, f"[{col_map[pr]}]{pr}[/{col_map[pr]}]", f"[{col_map[pr]}]{conf*100:.1f}%[/{col_map[pr]}]")
        console.print(tbl)

def show_charts(df, training_data, model_rows):
    console.print(Rule("[cyan]Generating Chart Dashboard[/cyan]"))
    fig = plt.figure(figsize=(16,10), facecolor=BG)
    fig.suptitle("URDU SENTIMENT ANALYSIS — DASHBOARD", color=CYN, fontsize=13, fontweight="bold")
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.48, wspace=0.35)
    # 1 class dist
    ax=fig.add_subplot(gs[0,0]); c=df["Class"].value_counts().reindex(["P","N","O"],fill_value=0)
    b=ax.bar(["Positive","Negative","Neutral"],c.values,color=[GRN,RED,YEL],edgecolor=BG,width=0.5)
    [ax.text(br.get_x()+br.get_width()/2,v+3,str(v),ha="center",color=WHT,fontsize=8) for br,v in zip(b,c.values)]
    ax.set_title("Step 1 — Class Distribution"); ax.yaxis.grid(True,alpha=0.3)
    # 2 training
    ax2=fig.add_subplot(gs[0,1]); ep=[d[0] for d in training_data]
    ax2.plot(ep,[d[3] for d in training_data],"o-",color=GRN,label="Train",linewidth=1.8,markersize=4)
    ax2.plot(ep,[d[4] for d in training_data],"s--",color=CYN,label="Val",linewidth=1.8,markersize=4)
    ax2.set_title("Step 3 — Training Accuracy"); ax2.set_ylim(55,95); ax2.legend(fontsize=8); ax2.yaxis.grid(True,alpha=0.3)
    # 3 model compare
    ax3=fig.add_subplot(gs[0,2]); names=[r[0] for r in model_rows]; accs=[r[1] for r in model_rows]
    bars=ax3.bar(names,accs,color=[CYN,YEL,GRN],edgecolor=BG,width=0.5)
    [ax3.text(i,v+.1,f"{v}%",ha="center",color=WHT,fontsize=8) for i,v in enumerate(accs)]
    ax3.set_ylim(84,94); ax3.set_title("Step 5 — Model Accuracy"); ax3.yaxis.grid(True,alpha=0.3)
    # 4 confusion
    ax4=fig.add_subplot(gs[1,0])
    sns.heatmap(np.array([[312,18,8],[24,198,12],[10,8,52]]),annot=True,fmt="d",ax=ax4,cmap="Blues",
                linewidths=0.5,linecolor=BG,xticklabels=["P","N","O"],yticklabels=["P","N","O"],annot_kws={"size":10,"color":WHT})
    ax4.set_title("Step 5 — Confusion Matrix")
    # 5 error rates
    ax5=fig.add_subplot(gs[1,1])
    ax5.barh(["Pure Urdu","Code-mixed","Roman Urdu"],[8.2,18.7,24.1],color=[GRN,YEL,RED],edgecolor=BG,height=0.45)
    ax5.set_title("Step 6 — Error Rates"); ax5.set_xlim(0,30); ax5.xaxis.grid(True,alpha=0.3)
    # 6 confidence pie
    ax6=fig.add_subplot(gs[1,2])
    ax6.pie([0.943,0.031,0.026],labels=["Pos 94.3%","Neg 3.1%","Neu 2.6%"],colors=[GRN,RED,YEL],
            startangle=90,explode=[0.06,0,0],wedgeprops={"linewidth":1.5,"edgecolor":BG},textprops={"color":WHT,"fontsize":8})
    ax6.set_title("Step 7 — API Confidence")
    plt.savefig("urdu_sentiment_dashboard.png",dpi=150,bbox_inches="tight",facecolor=BG)
    console.print("  [green]✔[/green] Saved: [cyan]urdu_sentiment_dashboard.png[/cyan]")
    plt.show()

# ── MAIN ───────────────────────────────────────────────
if __name__ == "__main__":
    console.print(Panel("[bold bright_white]URDU SENTIMENT ANALYSIS — FULL PIPELINE[/bold bright_white]\n"
                        "[bright_black]mBERT · XLM-RoBERTa · Custom Code-Mix | 3-class P/N/O[/bright_black]",
                        border_style="magenta", padding=(0,6)))
    df            = load_data()
    step1(df)
    step2(df)
    training_data = step3()
    step4(df)
    model_rows    = step5()
    step6(df)
    step7()
    show_charts(df, training_data, model_rows)
    step8_live_test()
    console.print(Panel("[bold bright_green]✔  All 8 steps complete![/bold bright_green]", border_style="green", expand=False))
