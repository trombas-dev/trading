//+------------------------------------------------------------------+
//|                                        Hermes_Structure.mq5     |
//|                        Hermes Trading  –  MQL5 custom indicator |
//+------------------------------------------------------------------+
//  Draws on the chart:
//
//  S/R LEVELS        – solid horizontal segment at every confirmed swing
//                      HIGH (R) and swing LOW (S), InpExtend bars wide.
//
//  BOS               – dashed line at a broken swing level when the close
//                      breaks in the direction of the established trend.
//
//  CHoCH             – same but breaks against the trend (reversal signal).
//
//  FAIR VALUE GAP    – filled rectangle between high[i-2] and low[i]
//                      (bullish) or between low[i-2] and high[i] (bearish)
//                      when a 3-bar imbalance exists.  Box extends
//                      InpExtend bars to the right of the gap bar.
//
//  LIQUIDITY SWEEP   – star (★) on a bar whose wick pierces the most
//                      recent swing HIGH / LOW but whose CLOSE retreats
//                      back inside (stop-hunt / fake-out pattern).
//
//  ORDER BLOCK       – filled zone at the last opposing candle before a
//                      BOS/CHoCH impulse (the candle where institutions
//                      are assumed to have placed orders).
//                        Bullish OB = last bearish candle before bullish BOS/CHoCH
//                        Bearish OB = last bullish candle before bearish BOS/CHoCH
//                      The box extends to the right until price CLOSES fully
//                      through the zone (mitigated → changes to grey).
//
//+------------------------------------------------------------------+
#property copyright   "Hermes Trading"
#property version     "1.20"
#property description "S/R  +  BOS/CHoCH  +  FVG  +  Sweep  +  Order Block"
#property indicator_chart_window
#property indicator_buffers 0
#property indicator_plots   0

//──────────────────────────────────────────────────────── Inputs ────
input group              "=== Swing detection ==="
input int    InpN              = 10;               // Bars each side for swing pivot
input int    InpExtend         = 5;                // Bars every drawn object extends right

input group              "=== S/R & Structure colours ==="
input color  InpResistColor    = clrTomato;        // Resistance  (swing HIGH)
input color  InpSupportColor   = clrCornflowerBlue;// Support     (swing LOW)
input color  InpBosBullColor   = clrLimeGreen;     // Bullish BOS
input color  InpBosBearColor   = clrCrimson;       // Bearish BOS
input color  InpChochBullColor = clrGold;          // Bullish CHoCH
input color  InpChochBearColor = clrDarkOrange;    // Bearish CHoCH

input group              "=== Fair Value Gap ==="
input bool   InpShowFVG        = true;             // Draw FVG boxes
input int    InpFVGMinPips     = 5;                // Min gap size in points (0 = off)
input double InpFVGMinBody     = 0.5;              // Min middle-candle body/range ratio (0 = off)
input color  InpFVGBullColor   = C'0,160,80';      // Bullish FVG fill colour
input color  InpFVGBearColor   = C'180,40,40';     // Bearish FVG fill colour

input group              "=== Liquidity Sweep ==="
input bool   InpShowSweep      = true;             // Draw liquidity sweep stars
input color  InpSweepBullColor = clrAqua;          // Bullish sweep (wick below SL)
input color  InpSweepBearColor = clrMagenta;       // Bearish sweep (wick above SH)

input group              "=== Order Block ==="
input bool   InpShowOB         = true;             // Draw Order Blocks
input bool   InpOBFullCandle   = false;            // true = full candle zone, false = body only
input int    InpOBLookback     = 20;               // Max bars to scan back for OB candle
input color  InpOBBullColor    = C'20,80,180';     // Bullish OB (fresh)
input color  InpOBBearColor    = C'180,50,20';     // Bearish OB (fresh)
input color  InpOBMitColor     = C'80,80,80';      // Mitigated OB

input group              "=== Style ==="
input int    InpLineWidth      = 2;                // S/R & BOS line width (1 – 5)
input bool   InpShowSR         = true;             // Draw S/R segments
input bool   InpShowStructure  = true;             // Draw BOS / CHoCH

//──────────────────────────────────────────────────── Internals ─────
#define PFX        "HMS_"
#define FAR_FUTURE  D'2038.01.01'   // right edge for active (unmitigated) OB boxes

struct SwingPt
{
   int      bar;
   double   price;
   datetime t;
};

// Order Block record — tracks zone and mitigation state
struct OBBlock
{
   double   zone_hi;
   double   zone_lo;
   bool     is_bull;
   bool     mitigated;
   string   box_nm;    // object names for colour update on mitigation
   string   lbl_nm;
};

// ── BOS / CHoCH state ──────────────────────────────────────────────
SwingPt  g_SH,     g_SL;      // latest unbroken swing HIGH / LOW
bool     g_hasSH,  g_hasSL;   // validity flags
int      g_trend;              // 1=up  -1=down  0=undef
int      g_lastSHBar, g_lastSLBar;   // last registered swing bar (dedup)

// ── Liquidity Sweep state ──────────────────────────────────────────
SwingPt  g_swpSH,  g_swpSL;
bool     g_hasSwpSH, g_hasSwpSL;

// ── Order Block registry ───────────────────────────────────────────
OBBlock  g_obs[];
int      g_obCount;

//+------------------------------------------------------------------+
void _Reset()
{
   ZeroMemory(g_SH);    ZeroMemory(g_SL);
   ZeroMemory(g_swpSH); ZeroMemory(g_swpSL);
   g_hasSH     = false; g_hasSL     = false;
   g_hasSwpSH  = false; g_hasSwpSL  = false;
   g_trend     = 0;
   g_lastSHBar = -1;    g_lastSLBar = -1;
   ArrayResize(g_obs, 0);
   g_obCount   = 0;
}

int OnInit()
{
   ObjectsDeleteAll(0, PFX);
   _Reset();
   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason)
{
   ObjectsDeleteAll(0, PFX);
}

//──────────────────────────────────────────── Drawing helpers ────────

// Returns the datetime InpExtend bars after bar[idx], extrapolating if needed.
datetime _T2(const datetime &t[], int idx, int total)
{
   int e = idx + InpExtend;
   if(e < total) return t[e];
   if(total < 2) return t[total - 1];
   long step = (long)(t[total - 1] - t[total - 2]);
   return t[total - 1] + (datetime)((e - total + 1) * step);
}

// Fixed horizontal segment.  Idempotent.
void _Seg(const string name, double price,
          datetime t1, datetime t2,
          color clr, int w,
          ENUM_LINE_STYLE sty = STYLE_SOLID,
          string tip = "")
{
   if(ObjectFind(0, name) >= 0) return;
   ObjectCreate(0, name, OBJ_TREND, 0, t1, price, t2, price);
   ObjectSetInteger(0, name, OBJPROP_COLOR,      clr);
   ObjectSetInteger(0, name, OBJPROP_WIDTH,      w);
   ObjectSetInteger(0, name, OBJPROP_RAY_RIGHT,  false);
   ObjectSetInteger(0, name, OBJPROP_RAY_LEFT,   false);
   ObjectSetInteger(0, name, OBJPROP_STYLE,      sty);
   ObjectSetInteger(0, name, OBJPROP_SELECTABLE, false);
   ObjectSetInteger(0, name, OBJPROP_HIDDEN,     true);
   ObjectSetInteger(0, name, OBJPROP_BACK,       false);
   if(tip != "") ObjectSetString(0, name, OBJPROP_TOOLTIP, tip);
}

// Text label.  Idempotent.
void _Lbl(const string name, const string txt,
          datetime t, double price, color clr, int sz = 8)
{
   if(ObjectFind(0, name) >= 0) return;
   ObjectCreate(0, name, OBJ_TEXT, 0, t, price);
   ObjectSetString (0, name, OBJPROP_TEXT,       txt);
   ObjectSetInteger(0, name, OBJPROP_COLOR,      clr);
   ObjectSetInteger(0, name, OBJPROP_FONTSIZE,   sz);
   ObjectSetString (0, name, OBJPROP_FONT,       "Arial Bold");
   ObjectSetInteger(0, name, OBJPROP_SELECTABLE, false);
   ObjectSetInteger(0, name, OBJPROP_HIDDEN,     true);
   ObjectSetInteger(0, name, OBJPROP_BACK,       false);
}

// Filled rectangle for FVG.  Idempotent.
void _Box(const string name,
          double price_hi, double price_lo,
          datetime t1,     datetime t2,
          color clr, string tip = "")
{
   if(ObjectFind(0, name) >= 0) return;
   ObjectCreate(0, name, OBJ_RECTANGLE, 0, t1, price_hi, t2, price_lo);
   ObjectSetInteger(0, name, OBJPROP_COLOR,      clr);
   ObjectSetInteger(0, name, OBJPROP_WIDTH,      1);
   ObjectSetInteger(0, name, OBJPROP_FILL,       true);
   ObjectSetInteger(0, name, OBJPROP_BACK,       true);   // render behind candles
   ObjectSetInteger(0, name, OBJPROP_SELECTABLE, false);
   ObjectSetInteger(0, name, OBJPROP_HIDDEN,     true);
   if(tip != "") ObjectSetString(0, name, OBJPROP_TOOLTIP, tip);
}

// Liquidity sweep star ★.  Idempotent.
// above=true  → star sits above the bar (bearish sweep over swing HIGH)
// above=false → star sits below the bar (bullish sweep under swing LOW)
void _Star(const string name, datetime t, double price,
           color clr, bool above)
{
   if(ObjectFind(0, name) >= 0) return;
   ObjectCreate(0, name, OBJ_TEXT, 0, t, price);
   ObjectSetString (0, name, OBJPROP_TEXT,       "★");
   ObjectSetInteger(0, name, OBJPROP_COLOR,      clr);
   ObjectSetInteger(0, name, OBJPROP_FONTSIZE,   14);
   ObjectSetString (0, name, OBJPROP_FONT,       "Arial");
   // ANCHOR_LOWER: text bottom anchored to price → text appears ABOVE → use for bear sweep
   // ANCHOR_UPPER: text top anchored to price    → text appears BELOW → use for bull sweep
   ObjectSetInteger(0, name, OBJPROP_ANCHOR,     above ? ANCHOR_LOWER : ANCHOR_UPPER);
   ObjectSetInteger(0, name, OBJPROP_SELECTABLE, false);
   ObjectSetInteger(0, name, OBJPROP_HIDDEN,     true);
   ObjectSetInteger(0, name, OBJPROP_BACK,       false);
   ObjectSetString (0, name, OBJPROP_TOOLTIP,    "Liquidity Sweep");
}

//──────────────────────────────────────── Swing-pivot detection ──────

bool _IsSH(const double &h[], int idx, int total)
{
   if(idx < InpN || idx + InpN >= total) return false;
   double p = h[idx];
   for(int k = idx - InpN; k <= idx + InpN; k++)
      if(k != idx && h[k] >= p) return false;
   return true;
}

bool _IsSL(const double &l[], int idx, int total)
{
   if(idx < InpN || idx + InpN >= total) return false;
   double p = l[idx];
   for(int k = idx - InpN; k <= idx + InpN; k++)
      if(k != idx && l[k] <= p) return false;
   return true;
}

//──────────────────────────────────── Order Block helpers ────────────

// Register and draw an Order Block, then append it to g_obs[].
void _RegisterOB(int ob_bar, bool is_bull, int break_bar,
                 const datetime &time[],
                 const double   &open[],
                 const double   &high[],
                 const double   &low[],
                 const double   &close[])
{
   // ── Zone boundaries ──────────────────────────────────────────────
   double zone_hi, zone_lo;
   if(InpOBFullCandle)
   {
      zone_hi = high[ob_bar];
      zone_lo = low[ob_bar];
   }
   else
   {
      // Body only:
      //   Bullish OB candle is bearish  → open (top) and close (bottom)
      //   Bearish OB candle is bullish  → close (top) and open (bottom)
      if(is_bull) { zone_hi = open[ob_bar];  zone_lo = close[ob_bar]; }
      else        { zone_hi = close[ob_bar]; zone_lo = open[ob_bar];  }
   }

   color  clr     = is_bull ? InpOBBullColor : InpOBBearColor;
   string sfx     = (is_bull ? "OBu_" : "OBd_") + IntegerToString(ob_bar);
   string box_nm  = PFX + sfx + "b";
   string lbl_nm  = PFX + sfx + "l";
   string dir_lbl = is_bull ? "Bull OB" : "Bear OB";

   // Box: from OB candle open-time → FAR_FUTURE (trimmed when mitigated)
   if(ObjectFind(0, box_nm) < 0)
   {
      ObjectCreate(0, box_nm, OBJ_RECTANGLE, 0,
                   time[ob_bar], zone_hi, FAR_FUTURE, zone_lo);
      ObjectSetInteger(0, box_nm, OBJPROP_COLOR,      clr);
      ObjectSetInteger(0, box_nm, OBJPROP_WIDTH,      1);
      ObjectSetInteger(0, box_nm, OBJPROP_FILL,       true);
      ObjectSetInteger(0, box_nm, OBJPROP_BACK,       true);
      ObjectSetInteger(0, box_nm, OBJPROP_SELECTABLE, false);
      ObjectSetInteger(0, box_nm, OBJPROP_HIDDEN,     true);
      ObjectSetString (0, box_nm, OBJPROP_TOOLTIP,
                       dir_lbl + "  " + DoubleToString(zone_lo, _Digits) +
                       " – "          + DoubleToString(zone_hi, _Digits));
   }

   // Label at the left edge of the box
   _Lbl(lbl_nm, dir_lbl,
        time[ob_bar],
        is_bull ? zone_hi + _Point * 3 : zone_lo - _Point * 3,
        clr, 8);

   // Append to registry for mitigation tracking
   ArrayResize(g_obs, g_obCount + 1);
   g_obs[g_obCount].zone_hi   = zone_hi;
   g_obs[g_obCount].zone_lo   = zone_lo;
   g_obs[g_obCount].is_bull   = is_bull;
   g_obs[g_obCount].mitigated = false;
   g_obs[g_obCount].box_nm    = box_nm;
   g_obs[g_obCount].lbl_nm    = lbl_nm;
   g_obCount++;
}

//+------------------------------------------------------------------+
//|  Main calculation                                                |
//+------------------------------------------------------------------+
int OnCalculate(const int      rates_total,
                const int      prev_calculated,
                const datetime &time[],
                const double   &open[],
                const double   &high[],
                const double   &low[],
                const double   &close[],
                const long     &tick_volume[],
                const long     &volume[],
                const int      &spread[])
{
   if(rates_total < 2 * InpN + 2) return 0;

   if(prev_calculated == 0)
   {
      ObjectsDeleteAll(0, PFX);
      _Reset();
   }

   // ════════════════════════════════════════════════════════════════
   //  PASS 1 – Fair Value Gap
   //  Runs from bar 2 (no swing-confirmation delay needed).
   //  A 3-bar pattern: bar[i-2], bar[i-1], bar[i].
   //    Bullish FVG: high[i-2] < low[i]   → gap between them
   //    Bearish FVG: low[i-2]  > high[i]  → gap between them
   // ════════════════════════════════════════════════════════════════
   if(InpShowFVG)
   {
      int fvg_start = (prev_calculated == 0) ? 2 : MathMax(prev_calculated - 2, 2);

      for(int i = fvg_start; i < rates_total; i++)
      {
         // ── Pre-compute middle candle body/range ratio ───────────────
         double mid_range = high[i-1] - low[i-1];
         double mid_body  = MathAbs(close[i-1] - open[i-1]);
         double mid_ratio = (mid_range > 0) ? mid_body / mid_range : 0.0;

         // ── Bullish FVG ──────────────────────────────────────────
         if(high[i-2] < low[i])
         {
            double gap  = low[i] - high[i-2];
            bool   ok   = true;

            if(InpFVGMinPips > 0 && gap < InpFVGMinPips * _Point) ok = false;  // gap too small
            if(InpFVGMinBody > 0 && mid_ratio < InpFVGMinBody)    ok = false;  // middle candle too indecisive
            if(close[i-1] < open[i-1])                            ok = false;  // middle candle must be bullish

            if(ok)
            {
               _Box(PFX + "FVGu_" + IntegerToString(i),
                    low[i], high[i-2],
                    time[i], _T2(time, i, rates_total),
                    InpFVGBullColor,
                    "Bull FVG  " + DoubleToString(high[i-2], _Digits) +
                    " – "        + DoubleToString(low[i],    _Digits));
            }
         }

         // ── Bearish FVG ──────────────────────────────────────────
         if(low[i-2] > high[i])
         {
            double gap  = low[i-2] - high[i];
            bool   ok   = true;

            if(InpFVGMinPips > 0 && gap < InpFVGMinPips * _Point) ok = false;  // gap too small
            if(InpFVGMinBody > 0 && mid_ratio < InpFVGMinBody)    ok = false;  // middle candle too indecisive
            if(close[i-1] > open[i-1])                            ok = false;  // middle candle must be bearish

            if(ok)
            {
               _Box(PFX + "FVGd_" + IntegerToString(i),
                    low[i-2], high[i],
                    time[i], _T2(time, i, rates_total),
                    InpFVGBearColor,
                    "Bear FVG  " + DoubleToString(high[i],   _Digits) +
                    " – "        + DoubleToString(low[i-2],  _Digits));
            }
         }
      }
   }

   // ════════════════════════════════════════════════════════════════
   //  PASS 2 – Swing S/R  +  BOS/CHoCH  +  Liquidity Sweep
   //  i iterates forward; swing pivot j = i - InpN is confirmed
   //  once we have InpN bars to its right.
   // ════════════════════════════════════════════════════════════════
   int start = (prev_calculated == 0)
               ? 2 * InpN
               : MathMax(prev_calculated - 1, 2 * InpN);

   for(int i = start; i < rates_total; i++)
   {
      int j = i - InpN;   // swing candidate bar

      // ── 1. Register newly confirmed swing HIGH ───────────────────
      if(j > g_lastSHBar && _IsSH(high, j, rates_total))
      {
         g_lastSHBar  = j;
         g_SH.bar     = j;  g_SH.price  = high[j];  g_SH.t  = time[j];
         g_hasSH      = true;
         g_swpSH      = g_SH;   // update sweep reference (never consumed)
         g_hasSwpSH   = true;

         if(InpShowSR)
         {
            _Seg(PFX + "R_"  + IntegerToString(j),
                 high[j], time[j], _T2(time, j, rates_total),
                 InpResistColor, InpLineWidth, STYLE_SOLID,
                 "R  " + DoubleToString(high[j], _Digits));
            _Lbl(PFX + "RL_" + IntegerToString(j), "R",
                 time[j], high[j] + _Point * 3, InpResistColor, 7);
         }
      }

      // ── 2. Register newly confirmed swing LOW ────────────────────
      if(j > g_lastSLBar && _IsSL(low, j, rates_total))
      {
         g_lastSLBar  = j;
         g_SL.bar     = j;  g_SL.price  = low[j];  g_SL.t  = time[j];
         g_hasSL      = true;
         g_swpSL      = g_SL;   // update sweep reference
         g_hasSwpSL   = true;

         if(InpShowSR)
         {
            _Seg(PFX + "S_"  + IntegerToString(j),
                 low[j], time[j], _T2(time, j, rates_total),
                 InpSupportColor, InpLineWidth, STYLE_SOLID,
                 "S  " + DoubleToString(low[j], _Digits));
            _Lbl(PFX + "SL_" + IntegerToString(j), "S",
                 time[j], low[j] - _Point * 3, InpSupportColor, 7);
         }
      }

      // ── 3. Order Block — mitigation check ───────────────────────
      //  A bullish OB is mitigated when close drops BELOW zone_lo.
      //  A bearish OB is mitigated when close rises ABOVE zone_hi.
      //  On mitigation: trim the box right edge to this bar and recolour.
      if(InpShowOB)
      {
         for(int k = 0; k < g_obCount; k++)
         {
            if(g_obs[k].mitigated) continue;

            bool mit = (g_obs[k].is_bull  && close[i] < g_obs[k].zone_lo)
                    || (!g_obs[k].is_bull && close[i] > g_obs[k].zone_hi);
            if(mit)
            {
               g_obs[k].mitigated = true;
               // Trim right edge to mitigation bar
               ObjectSetInteger(0, g_obs[k].box_nm, OBJPROP_TIME,  1, time[i]);
               // Recolour to mitigated shade
               ObjectSetInteger(0, g_obs[k].box_nm, OBJPROP_COLOR, InpOBMitColor);
               ObjectSetInteger(0, g_obs[k].lbl_nm, OBJPROP_COLOR, InpOBMitColor);
            }
         }
      }

      // ── 4. Liquidity Sweep detection ─────────────────────────────
      //  Bearish sweep: wick punches ABOVE last swing HIGH,
      //                 close retreats BACK BELOW it  → stop hunt on longs
      //  Bullish sweep: wick punches BELOW last swing LOW,
      //                 close retreats BACK ABOVE it  → stop hunt on shorts
      if(InpShowSweep)
      {
         if(g_hasSwpSH
            && high[i]  > g_swpSH.price
            && close[i] < g_swpSH.price)
         {
            _Star(PFX + "SWb_" + IntegerToString(i),
                  time[i], high[i], InpSweepBearColor, true);
         }

         if(g_hasSwpSL
            && low[i]   < g_swpSL.price
            && close[i] > g_swpSL.price)
         {
            _Star(PFX + "SWu_" + IntegerToString(i),
                  time[i], low[i], InpSweepBullColor, false);
         }
      }

      if(!InpShowStructure) continue;

      // ── 5. Bullish break – BOS or CHoCH ──────────────────────────
      if(g_hasSH && close[i] > g_SH.price)
      {
         string nm = PFX + "BSu_" + IntegerToString(i);
         if(ObjectFind(0, nm) < 0)
         {
            bool   isBOS = (g_trend == 1);
            color  clr   = isBOS ? InpBosBullColor   : InpChochBullColor;
            string lbl   = isBOS ? "BOS"             : "CHoCH";
            _Seg(nm, g_SH.price,
                 g_SH.t, _T2(time, i, rates_total),
                 clr, InpLineWidth + 1, STYLE_DASH,
                 lbl + " (bull)  " + DoubleToString(g_SH.price, _Digits));
            _Lbl(PFX + "BLu_" + IntegerToString(i), lbl,
                 time[i], g_SH.price + _Point * 3, clr, 9);
            g_trend = 1;
            g_hasSH = false;

            // ── Find & register the bullish Order Block ──────────────
            if(InpShowOB)
            {
               // Scan back for the last BEARISH candle before bar i
               int ob_bar = -1;
               int limit  = MathMax(0, i - 1 - InpOBLookback);
               for(int k = i - 1; k >= limit; k--)
               {
                  if(close[k] < open[k]) { ob_bar = k; break; }
               }
               if(ob_bar >= 0)
                  _RegisterOB(ob_bar, true, i, time, open, high, low, close);
            }
         }
      }

      // ── 6. Bearish break – BOS or CHoCH ──────────────────────────
      if(g_hasSL && close[i] < g_SL.price)
      {
         string nm = PFX + "BSd_" + IntegerToString(i);
         if(ObjectFind(0, nm) < 0)
         {
            bool   isBOS = (g_trend == -1);
            color  clr   = isBOS ? InpBosBearColor   : InpChochBearColor;
            string lbl   = isBOS ? "BOS"             : "CHoCH";
            _Seg(nm, g_SL.price,
                 g_SL.t, _T2(time, i, rates_total),
                 clr, InpLineWidth + 1, STYLE_DASH,
                 lbl + " (bear)  " + DoubleToString(g_SL.price, _Digits));
            _Lbl(PFX + "BLd_" + IntegerToString(i), lbl,
                 time[i], g_SL.price - _Point * 3, clr, 9);
            g_trend = -1;
            g_hasSL = false;

            // ── Find & register the bearish Order Block ──────────────
            if(InpShowOB)
            {
               int ob_bar = -1;
               int limit  = MathMax(0, i - 1 - InpOBLookback);
               for(int k = i - 1; k >= limit; k--)
               {
                  if(close[k] > open[k]) { ob_bar = k; break; }
               }
               if(ob_bar >= 0)
                  _RegisterOB(ob_bar, false, i, time, open, high, low, close);
            }
         }
      }
   }

   return rates_total;
}
//+------------------------------------------------------------------+
