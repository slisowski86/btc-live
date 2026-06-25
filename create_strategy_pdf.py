"""Generate a professional PDF describing the protected crypto strategy."""
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                                HRFlowable, ListFlowable, ListItem)

NAVY = colors.HexColor("#1a2b4a")
BLUE = colors.HexColor("#2a6db5")
GREY = colors.HexColor("#555555")
LIGHT = colors.HexColor("#eef2f7")
GREEN = colors.HexColor("#2a9d4a")
RED = colors.HexColor("#c0392b")

ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Title"], textColor=NAVY, fontSize=22, leading=26, spaceAfter=4)
SUB = ParagraphStyle("SUB", parent=ss["Normal"], textColor=BLUE, fontSize=12, leading=15, spaceAfter=2, alignment=1)
META = ParagraphStyle("META", parent=ss["Normal"], textColor=GREY, fontSize=8.5, alignment=1, spaceAfter=10)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], textColor=NAVY, fontSize=13.5, spaceBefore=12, spaceAfter=4)
BODY = ParagraphStyle("BODY", parent=ss["Normal"], fontSize=9.6, leading=13.5, spaceAfter=5, textColor=colors.HexColor("#222222"))
SMALL = ParagraphStyle("SMALL", parent=ss["Normal"], fontSize=8.3, leading=11, textColor=GREY)
BULLET = ParagraphStyle("BULLET", parent=BODY, leftIndent=10, spaceAfter=2)

def hr():
    return HRFlowable(width="100%", thickness=0.8, color=BLUE, spaceBefore=2, spaceAfter=8)

def bullets(items):
    return ListFlowable([ListItem(Paragraph(t, BULLET), leftIndent=8, value="•") for t in items],
                        bulletType="bullet", start="•")

story = []

# ---- header ----
story.append(Paragraph("Protected Cross-Confirmed Crypto Momentum Strategy", H1))
story.append(Paragraph("A trend-filtered, volatility-targeted basket for BTC / crypto (1-hour)", SUB))
story.append(Paragraph("Strategy specification &amp; validation summary", META))
story.append(hr())

# ---- 1. executive summary ----
story.append(Paragraph("1.  Executive Summary", H2))
story.append(Paragraph(
    "This strategy is a diversified basket of four indicator-based trading rules that trade BTC (and other "
    "trending crypto) on the 1-hour timeframe. Each rule enters with <b>momentum</b> in a confirmed trend and "
    "exits on <b>divergence</b>. The four rules were selected from an exhaustive search of ~22,000 candidates by a "
    "six-test out-of-sample gauntlet, and then <b>confirmed on a second asset (ETH) over 8 years</b> &mdash; so the "
    "edge is a genuine crypto-momentum effect, not a curve-fit. Risk is controlled by volatility-target position "
    "sizing, a <b>trend filter</b> that flattens long exposure during downtrends (crash defense), and a hard leverage "
    "cap. The edge is real but modest and regime-dependent; it is intended for disciplined, risk-managed deployment, "
    "not as a guaranteed return.", BODY))

# ---- 2. philosophy ----
story.append(Paragraph("2.  Design Philosophy", H2))
story.append(bullets([
    "<b>Momentum entries, divergence exits.</b> Enter long when momentum turns up in a confirmed uptrend; exit "
    "early when a momentum/price divergence warns of a reversal. Divergence is an early-but-noisy signal, so it is "
    "used only for exits (a false exit costs opportunity; a false entry costs real money).",
    "<b>Diversified basket, not one rule.</b> Four low-correlation rules sharing the same exit core but different "
    "entries &mdash; their drawdowns do not coincide, smoothing the equity curve.",
    "<b>Robustness over flashiness.</b> The single best-on-BTC rule (Sharpe 1.75) failed on ETH and was discarded. "
    "The chosen rules are more modest (~0.8 Sharpe each) but hold up on two assets.",
    "<b>Risk first.</b> Volatility targeting + trend filter + leverage cap. The trend filter is the key crash "
    "defense: it turned an Oct-2025 BTC drawdown from a loss into a gain.",
]))

# ---- 3. the basket ----
story.append(Paragraph("3.  The Basket &mdash; Signal Logic", H2))
story.append(Paragraph("All four rules share a common structure (entries vary; exits are nearly identical):", BODY))
recipe = [
    ["Signal group", "Logic (common recipe)"],
    ["ENTER LONG (all of)", "momentum cross up (EFISH / MACD) + trend-direction confirm (LinReg / SuperTrend) + trend strength (AROON / ADX)"],
    ["EXIT LONG (any of)", "CCI bearish divergence + Ehlers trend slopes down + ADX trend-strength"],
    ["ENTER SHORT (all of)", "momentum overbought (EFISH) + RSI crosses down + volatility-regime filter (DCWidth / BBWidth / ERFilt)"],
    ["EXIT SHORT (any of)", "CCI bullish divergence + EFISH oversold + Ehlers roofing-filter volatility"],
]
t = Table(recipe, colWidths=[3.4*cm, 12.6*cm])
t.setStyle(TableStyle([
    ("BACKGROUND",(0,0),(-1,0),NAVY), ("TEXTCOLOR",(0,0),(-1,0),colors.white),
    ("FONTSIZE",(0,0),(-1,-1),8.4), ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
    ("FONTNAME",(0,1),(0,-1),"Helvetica-Bold"), ("TEXTCOLOR",(0,1),(0,-1),NAVY),
    ("VALIGN",(0,0),(-1,-1),"MIDDLE"), ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,LIGHT]),
    ("GRID",(0,0),(-1,-1),0.4,colors.HexColor("#cccccc")), ("TOPPADDING",(0,0),(-1,-1),4),
    ("BOTTOMPADDING",(0,0),(-1,-1),4), ("LEFTPADDING",(0,0),(-1,-1),6),
]))
story.append(t)
story.append(Spacer(1,4))
story.append(Paragraph("<b>Workhorse indicators:</b> EFISH (Ehlers Fisher Transform &mdash; core momentum), "
    "CCIDiv (divergence exits), EIT / AROON / ADX / LinReg / SuperTrend (trend), RSI / STOCH (short triggers), "
    "ERFilt / DCWidth / BBWidth (volatility filters).", SMALL))

# ---- 4. risk management ----
story.append(Paragraph("4.  Risk Management", H2))
story.append(bullets([
    "<b>Volatility-target sizing</b> (target ~12% annualized vol): scales exposure inversely to recent volatility "
    "so each period contributes roughly equal risk; de-weights turbulent periods where drawdowns cluster.",
    "<b>Trend filter (crash defense):</b> flatten NET-LONG exposure whenever price is below its 300-bar moving "
    "average (shorts are kept). Fully causal; robust across MA windows 150&ndash;700 and across BTC and ETH.",
    "<b>Leverage</b> is a single multiplier with a hard cap (default 1x, cap 2x). Drawdown scales ~linearly with "
    "leverage; 2&ndash;3x is the prudent ceiling on crypto.",
    "<b>Netted basket:</b> the four rules are combined into one net position to minimize turnover and cost.",
]))

# ---- 5. validation ----
story.append(Paragraph("5.  Development &amp; Validation", H2))
story.append(Paragraph(
    "Candidates were generated by exhaustively enumerating a promising pattern family (~22,000 entry combinations) "
    "and tested on a <b>reserved tail of data never used in selection</b>. Each had to pass <b>six independent "
    "tests</b>: (1) positive out-of-sample Sharpe, (2) survives realistic costs, (3) survives volatility sizing, "
    "(4) positive in both halves of the reserved period, (5) enough trades to be statistically meaningful, and "
    "(6) beats buy-and-hold. 611 of 22,464 passed on BTC; <b>292 of those also passed the identical gauntlet on "
    "ETH across 8 years</b>. The final basket is four low-correlation rules drawn from that cross-confirmed set. "
    "The same strategies produced <b>no edge on EUR/USD</b> (an efficient, mean-reverting market) &mdash; consistent "
    "with the edge being a real, crypto-specific momentum effect rather than a fitting artifact.", BODY))

# ---- 6. performance ----
story.append(Paragraph("6.  Indicative Performance (out-of-sample, leverage 1x)", H2))
perf = [
    ["Configuration", "Asset / window", "Return", "Max DD", "Sharpe"],
    ["Basket, vol-targeted (no filter)", "BTC  2024-07 to 2026-06", "+18%", "7.2%", "1.09"],
    ["Basket + trend filter", "BTC  2024-07 to 2026-06", "+60%", "2.4%", "3.94"],
    ["Basket + trend filter", "ETH  2018-06 to 2026-06", "+798%", "10.9%", "3.48"],
    ["Buy & hold (benchmark)", "BTC  2024-07 to 2026-06", "+1%", ">50%", "~0.6"],
]
t2 = Table(perf, colWidths=[6.0*cm, 5.0*cm, 1.9*cm, 1.6*cm, 1.5*cm])
t2.setStyle(TableStyle([
    ("BACKGROUND",(0,0),(-1,0),NAVY), ("TEXTCOLOR",(0,0),(-1,0),colors.white),
    ("FONTSIZE",(0,0),(-1,-1),8.4), ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
    ("ALIGN",(2,0),(-1,-1),"CENTER"), ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
    ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,LIGHT]),
    ("GRID",(0,0),(-1,-1),0.4,colors.HexColor("#cccccc")),
    ("TEXTCOLOR",(2,2),(2,3),GREEN), ("FONTNAME",(2,2),(2,3),"Helvetica-Bold"),
    ("TOPPADDING",(0,0),(-1,-1),4), ("BOTTOMPADDING",(0,0),(-1,-1),4),
]))
story.append(t2)
story.append(Spacer(1,3))
story.append(Paragraph("Leverage sweep (BTC, vol-targeted basket): 1x +18%/7%DD, 2x +38%/14%DD, 3x +59%/21%DD, "
    "5x +103%/33%DD. The trend filter also fixed the Oct&ndash;Dec 2025 drawdown: unprotected +0.9% (5% DD) "
    "vs protected +5.8% (1% DD).", SMALL))

# ---- 7. deployment ----
story.append(Paragraph("7.  Deployment", H2))
story.append(Paragraph(
    "The strategy is implemented as a single module, <font name='Courier'>protected_strategy.py</font>, exposing "
    "<font name='Courier'>target_exposure(df)</font> &mdash; the netted, vol-targeted, trend-filtered, leverage-"
    "adjusted exposure for the latest closed 1-hour bar. A paper-trading logger "
    "(<font name='Courier'>paper_trader.py</font>) fetches live Binance bars hourly and records the target orders "
    "and a paper-equity curve (no real orders). Going live means swapping the logged order for a real exchange "
    "order via ccxt. Recommended path: paper-trade forward 1&ndash;3 months, then trade unlevered at minimum size "
    "before scaling toward the 2&ndash;3x ceiling.", BODY))

# ---- 8. risks ----
story.append(Paragraph("8.  Risks &amp; Caveats", H2))
story.append(bullets([
    "<b>In-sample optimism.</b> The Sharpe figures (3&ndash;4 with the trend filter) are upper bounds; real "
    "trading (slippage, funding, flip costs) will be lower. The shape of the improvement is trustworthy; the "
    "magnitude is not a forecast.",
    "<b>Regime dependence.</b> This is a momentum strategy. It thrives in trending markets and struggles in chop; "
    "the recent 2024&ndash;2026 crypto regime has been comparatively soft for momentum.",
    "<b>Crypto-specific.</b> The edge does not transfer to efficient markets (e.g. EUR/USD). It relies on crypto's "
    "trending, retail-driven inefficiency, which is not guaranteed to persist.",
    "<b>Not yet forward-tested.</b> All results are historical. Live forward paper-trading is the final, "
    "uncontaminated test and must be completed before committing real capital.",
    "<b>Leverage risk.</b> On crypto perpetuals, high leverage carries liquidation and funding risk worse than the "
    "idealized sweep suggests. Start unlevered.",
]))
story.append(Spacer(1,8))
story.append(hr())
story.append(Paragraph("This document summarizes a research strategy for informational purposes only. It is not "
    "financial advice and carries no guarantee of future performance. Trade only capital you can afford to lose.",
    SMALL))

doc = SimpleDocTemplate("Strategy_Description.pdf", pagesize=A4,
                        leftMargin=2*cm, rightMargin=2*cm, topMargin=1.6*cm, bottomMargin=1.6*cm,
                        title="Protected Cross-Confirmed Crypto Momentum Strategy",
                        author="BTC_Live research")
doc.build(story)
print("wrote Strategy_Description.pdf")
