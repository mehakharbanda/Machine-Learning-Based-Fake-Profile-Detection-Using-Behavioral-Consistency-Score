import { useState, useEffect, useRef } from "react";

// ── Color palette ─────────────────────────────────────────────────────────
const C = {
  bg:       "#0a0c14",
  surface:  "#12151f",
  border:   "#1e2235",
  accent:   "#6c63ff",
  accentLt: "#a89cff",
  green:    "#2de37a",
  red:      "#ff4d6a",
  yellow:   "#f5c842",
  text:     "#dde3f0",
  muted:    "#6b7595",
};

// ── Radar chart for sub-scores ────────────────────────────────────────────
function RadarChart({ scores }) {
  const size = 180;
  const cx = size / 2, cy = size / 2, r = 72;
  const labels = ["Posting", "Engagement", "Profile", "Content", "Anti-spam"];
  const keys   = ["posting_regularity","engagement_authenticity","profile_completeness","content_quality","spam_signal"];
  const n = labels.length;

  const pts = (factor = 1) =>
    keys.map((k, i) => {
      const angle = (Math.PI * 2 * i) / n - Math.PI / 2;
      const val   = scores[k] ?? 0;
      return [cx + Math.cos(angle) * r * val * factor, cy + Math.sin(angle) * r * val * factor];
    });

  const polyStr = (pts) => pts.map(([x, y]) => `${x},${y}`).join(" ");
  const grid    = [0.25, 0.5, 0.75, 1].map(f => polyStr(pts(f)));
  const data    = polyStr(pts(1));
  const axes    = pts(1).map(([x, y], i) => {
    const ax = (Math.PI * 2 * i) / n - Math.PI / 2;
    const lx = cx + Math.cos(ax) * (r + 22);
    const ly = cy + Math.sin(ax) * (r + 22);
    return { x, y, lx, ly, label: labels[i] };
  });

  return (
    <svg viewBox={`0 0 ${size} ${size}`} width={size} height={size}>
      {grid.map((d, i) => (
        <polygon key={i} points={d} fill="none" stroke={C.border} strokeWidth="0.8" />
      ))}
      {axes.map(({ x, y, lx, ly, label }, i) => (
        <g key={i}>
          <line x1={cx} y1={cy} x2={x} y2={y} stroke={C.border} strokeWidth="0.8" />
          <text x={lx} y={ly} textAnchor="middle" dominantBaseline="middle"
                fill={C.muted} fontSize="9" fontFamily="'IBM Plex Mono', monospace">{label}</text>
        </g>
      ))}
      <polygon points={data} fill={C.accent + "33"} stroke={C.accent} strokeWidth="1.5" />
      {pts(1).map(([x, y], i) => (
        <circle key={i} cx={x} cy={y} r="3" fill={C.accent} />
      ))}
    </svg>
  );
}

// ── Score gauge ────────────────────────────────────────────────────────────
function BCSGauge({ score }) {
  const pct   = score / 100;
  const R     = 58, cx = 70, cy = 70;
  const circ  = 2 * Math.PI * R;
  const dash  = pct * circ * 0.75;
  const color = score >= 70 ? C.green : score >= 45 ? C.yellow : C.red;
  const label = score >= 70 ? "Genuine" : score >= 45 ? "Suspicious" : "Likely Fake";

  return (
    <div style={{ textAlign: "center" }}>
      <svg viewBox="0 0 140 100" width="160">
        <circle cx={cx} cy={cy} r={R} fill="none" stroke={C.border} strokeWidth="10"
                strokeDasharray={`${circ * 0.75} ${circ}`} strokeDashoffset={circ * 0.125}
                strokeLinecap="round" style={{ transform: "rotate(135deg)", transformOrigin: `${cx}px ${cy}px` }} />
        <circle cx={cx} cy={cy} r={R} fill="none" stroke={color} strokeWidth="10"
                strokeDasharray={`${dash} ${circ}`} strokeDashoffset={circ * 0.125}
                strokeLinecap="round" style={{ transform: "rotate(135deg)", transformOrigin: `${cx}px ${cy}px`,
                filter: `drop-shadow(0 0 6px ${color})`, transition: "stroke-dasharray 0.6s ease" }} />
        <text x={cx} y={cy - 4} textAnchor="middle" fill={color} fontSize="22" fontWeight="700"
              fontFamily="'IBM Plex Mono', monospace">{score.toFixed(0)}</text>
        <text x={cx} y={cy + 14} textAnchor="middle" fill={C.muted} fontSize="9"
              fontFamily="'IBM Plex Mono', monospace">BCS SCORE</text>
      </svg>
      <div style={{ color, fontWeight: 700, fontSize: 13, marginTop: -8, letterSpacing: "0.1em",
                    fontFamily: "'IBM Plex Mono', monospace" }}>{label}</div>
    </div>
  );
}

// ── Slider input ───────────────────────────────────────────────────────────
function SliderField({ label, name, value, min, max, step = 0.01, onChange }) {
  const pct = ((value - min) / (max - min)) * 100;
  return (
    <div style={{ marginBottom: 12 }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
        <span style={{ color: C.muted, fontSize: 11, fontFamily: "'IBM Plex Mono',monospace" }}>{label}</span>
        <span style={{ color: C.accentLt, fontSize: 11, fontFamily: "'IBM Plex Mono',monospace",
                       background: C.border, padding: "1px 6px", borderRadius: 4 }}>{value}</span>
      </div>
      <input type="range" min={min} max={max} step={step} value={value}
             onChange={e => onChange(name, parseFloat(e.target.value))}
             style={{ width: "100%", accentColor: C.accent, height: 3, cursor: "pointer" }} />
    </div>
  );
}

// ── Toggle field ───────────────────────────────────────────────────────────
function ToggleField({ label, name, value, onChange }) {
  return (
    <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between",
                  marginBottom: 12 }}>
      <span style={{ color: C.muted, fontSize: 11, fontFamily: "'IBM Plex Mono',monospace" }}>{label}</span>
      <div onClick={() => onChange(name, value ? 0 : 1)}
           style={{ width: 40, height: 20, borderRadius: 10, cursor: "pointer", transition: "background 0.2s",
                    background: value ? C.accent : C.border, position: "relative" }}>
        <div style={{ position: "absolute", top: 3, left: value ? 22 : 3, width: 14, height: 14,
                      borderRadius: "50%", background: "white", transition: "left 0.2s" }} />
      </div>
    </div>
  );
}

// ── Metric card ────────────────────────────────────────────────────────────
function MetricCard({ label, value, color }) {
  return (
    <div style={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 10,
                  padding: "12px 16px", flex: 1, minWidth: 90 }}>
      <div style={{ color: C.muted, fontSize: 10, fontFamily: "'IBM Plex Mono',monospace",
                    letterSpacing: "0.08em", marginBottom: 6 }}>{label}</div>
      <div style={{ color: color || C.text, fontSize: 20, fontWeight: 700,
                    fontFamily: "'IBM Plex Mono',monospace" }}>{value}</div>
    </div>
  );
}

// ── Bar for sub-scores ─────────────────────────────────────────────────────
function SubScoreBar({ label, value }) {
  const color = value >= 0.7 ? C.green : value >= 0.45 ? C.yellow : C.red;
  return (
    <div style={{ marginBottom: 10 }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
        <span style={{ color: C.muted, fontSize: 10, fontFamily: "'IBM Plex Mono',monospace" }}>{label}</span>
        <span style={{ color, fontSize: 10, fontFamily: "'IBM Plex Mono',monospace" }}>{(value * 100).toFixed(1)}%</span>
      </div>
      <div style={{ height: 6, background: C.border, borderRadius: 3, overflow: "hidden" }}>
        <div style={{ height: "100%", width: `${value * 100}%`, background: color, borderRadius: 3,
                      transition: "width 0.4s ease", boxShadow: `0 0 8px ${color}88` }} />
      </div>
    </div>
  );
}

// ── DEFAULT PROFILE VALUES ─────────────────────────────────────────────────
const DEFAULT_PROFILE = {
  posting_frequency: 2.5,
  follower_following_ratio: 2.0,
  account_age_days: 400,
  avg_likes_per_post: 80,
  avg_comments_per_post: 8,
  bio_completeness: 0.7,
  profile_pic_present: 1,
  url_in_bio: 0,
  verified: 0,
  posting_time_variance: 6,
  avg_post_length: 130,
  hashtag_ratio: 3,
  mention_ratio: 1,
  reply_consistency: 0.6,
  content_diversity_score: 0.65,
};

// ── BCS computation (client-side mirror) ──────────────────────────────────
function computeBCS(p) {
  const freqNorm  = Math.min(p.posting_frequency, 30) / 30;
  const freqScore = 1 - Math.abs(freqNorm - 0.15);
  const varNorm   = Math.min(p.posting_time_variance, 24) / 24;
  const postReg   = Math.max(0, Math.min(1, 0.5 * freqScore + 0.5 * varNorm));

  const likesN    = Math.log1p(p.avg_likes_per_post)    / Math.log1p(10000);
  const commN     = Math.log1p(p.avg_comments_per_post) / Math.log1p(1000);
  const ratioN    = Math.log1p(Math.min(p.follower_following_ratio, 50)) / Math.log1p(50);
  const eng       = (likesN + commN) / 2;
  const engAuth   = Math.max(0, Math.min(1, eng * 0.6 + (1 - Math.abs(eng - ratioN * 0.8)) * 0.4));

  const ageN      = Math.log1p(Math.min(p.account_age_days, 3650)) / Math.log1p(3650);
  const profComp  = Math.max(0, Math.min(1,
    p.bio_completeness * 0.35 + p.profile_pic_present * 0.25 + ageN * 0.30 + p.verified * 0.10
  ));

  const lenN      = Math.min(p.avg_post_length, 500) / 500;
  const lenScore  = 1 - Math.abs(lenN - 0.28);
  const contQual  = Math.max(0, Math.min(1,
    lenScore * 0.30 + p.content_diversity_score * 0.40 + p.reply_consistency * 0.30
  ));

  const hashN     = Math.min(p.hashtag_ratio, 15) / 15;
  const mentN     = Math.min(p.mention_ratio, 8) / 8;
  const spam      = Math.max(0, Math.min(1,
    1 - (hashN * 0.40 + mentN * 0.40 + p.url_in_bio * 0.20)
  ));

  const bcs_raw = postReg * 0.20 + engAuth * 0.25 + profComp * 0.20 + contQual * 0.20 + spam * 0.15;

  return {
    bcs_score:                 bcs_raw * 100,
    posting_regularity:        postReg,
    engagement_authenticity:   engAuth,
    profile_completeness:      profComp,
    content_quality:           contQual,
    spam_signal:               spam,
    is_fake:                   bcs_raw < 0.45,
    probability:               Math.max(0, Math.min(1, 1 - bcs_raw * 1.1)),
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// MAIN COMPONENT
// ─────────────────────────────────────────────────────────────────────────────
export default function App() {
  const [profile, setProfile] = useState(DEFAULT_PROFILE);
  const [result,  setResult]  = useState(null);
  const [loading, setLoading] = useState(false);
  const [tab,     setTab]     = useState("input");   // "input" | "result" | "about"

  // Compute live on every change
  useEffect(() => {
    const r = computeBCS(profile);
    setResult(r);
  }, [profile]);

  const handleChange = (name, val) =>
    setProfile(prev => ({ ...prev, [name]: val }));

  const riskColor = result
    ? result.bcs_score >= 70 ? C.green : result.bcs_score >= 45 ? C.yellow : C.red
    : C.muted;

  // ── Layout ──────────────────────────────────────────────────────────────
  return (
    <div style={{ background: C.bg, minHeight: "100vh", fontFamily: "'IBM Plex Mono', monospace",
                  color: C.text, padding: "0 0 40px" }}>

      {/* ── Header ── */}
      <div style={{ background: C.surface, borderBottom: `1px solid ${C.border}`,
                    padding: "16px 24px", display: "flex", alignItems: "center",
                    justifyContent: "space-between", position: "sticky", top: 0, zIndex: 10 }}>
        <div>
          <div style={{ fontSize: 15, fontWeight: 700, color: C.accentLt, letterSpacing: "0.05em" }}>
            FAKE PROFILE DETECTOR
          </div>
          <div style={{ fontSize: 10, color: C.muted, marginTop: 2, letterSpacing: "0.1em" }}>
            ML-BASED · BEHAVIORAL CONSISTENCY SCORE
          </div>
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          {["input", "result", "about"].map(t => (
            <button key={t} onClick={() => setTab(t)}
                    style={{ padding: "6px 14px", borderRadius: 6, cursor: "pointer", fontSize: 10,
                             fontFamily: "'IBM Plex Mono',monospace", letterSpacing: "0.08em",
                             border: `1px solid ${tab === t ? C.accent : C.border}`,
                             background: tab === t ? C.accent + "22" : "transparent",
                             color: tab === t ? C.accentLt : C.muted }}>
              {t.toUpperCase()}
            </button>
          ))}
        </div>
      </div>

      {/* ── Live BCS banner ── */}
      {result && (
        <div style={{ background: riskColor + "15", borderBottom: `1px solid ${riskColor}44`,
                      padding: "10px 24px", display: "flex", alignItems: "center", gap: 16 }}>
          <div style={{ width: 8, height: 8, borderRadius: "50%", background: riskColor,
                        boxShadow: `0 0 8px ${riskColor}` }} />
          <span style={{ color: riskColor, fontWeight: 700, fontSize: 12 }}>
            {result.bcs_score >= 70 ? "PROFILE APPEARS GENUINE" :
             result.bcs_score >= 45 ? "SUSPICIOUS ACTIVITY DETECTED" : "LIKELY FAKE PROFILE"}
          </span>
          <span style={{ color: C.muted, fontSize: 11 }}>
            BCS: {result.bcs_score.toFixed(1)} / 100 · Fake probability: {(result.probability * 100).toFixed(1)}%
          </span>
        </div>
      )}

      <div style={{ maxWidth: 900, margin: "0 auto", padding: "24px 20px" }}>

        {/* ══════════════════ INPUT TAB ══════════════════ */}
        {tab === "input" && (
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20 }}>

            {/* Left: Activity */}
            <div style={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 12, padding: 20 }}>
              <div style={{ color: C.accentLt, fontSize: 11, marginBottom: 16, letterSpacing: "0.1em" }}>
                📊 ACTIVITY METRICS
              </div>
              <SliderField label="Posting Frequency (posts/day)" name="posting_frequency"
                           value={profile.posting_frequency} min={0} max={30} step={0.1} onChange={handleChange} />
              <SliderField label="Posting Time Variance (hrs)" name="posting_time_variance"
                           value={profile.posting_time_variance} min={0} max={24} step={0.1} onChange={handleChange} />
              <SliderField label="Follower / Following Ratio" name="follower_following_ratio"
                           value={profile.follower_following_ratio} min={0} max={20} step={0.1} onChange={handleChange} />
              <SliderField label="Account Age (days)" name="account_age_days"
                           value={profile.account_age_days} min={1} max={3650} step={1} onChange={handleChange} />
              <SliderField label="Avg Likes per Post" name="avg_likes_per_post"
                           value={profile.avg_likes_per_post} min={0} max={5000} step={1} onChange={handleChange} />
              <SliderField label="Avg Comments per Post" name="avg_comments_per_post"
                           value={profile.avg_comments_per_post} min={0} max={500} step={1} onChange={handleChange} />
            </div>

            {/* Right: Content & Profile */}
            <div style={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 12, padding: 20 }}>
              <div style={{ color: C.accentLt, fontSize: 11, marginBottom: 16, letterSpacing: "0.1em" }}>
                🧩 CONTENT & PROFILE
              </div>
              <SliderField label="Bio Completeness (0-1)" name="bio_completeness"
                           value={profile.bio_completeness} min={0} max={1} step={0.01} onChange={handleChange} />
              <SliderField label="Avg Post Length (chars)" name="avg_post_length"
                           value={profile.avg_post_length} min={1} max={500} step={1} onChange={handleChange} />
              <SliderField label="Hashtag Ratio (per post)" name="hashtag_ratio"
                           value={profile.hashtag_ratio} min={0} max={20} step={0.1} onChange={handleChange} />
              <SliderField label="Mention Ratio (per post)" name="mention_ratio"
                           value={profile.mention_ratio} min={0} max={10} step={0.1} onChange={handleChange} />
              <SliderField label="Reply Consistency (0-1)" name="reply_consistency"
                           value={profile.reply_consistency} min={0} max={1} step={0.01} onChange={handleChange} />
              <SliderField label="Content Diversity (0-1)" name="content_diversity_score"
                           value={profile.content_diversity_score} min={0} max={1} step={0.01} onChange={handleChange} />
              <div style={{ display: "flex", gap: 16, marginTop: 4 }}>
                <div style={{ flex: 1 }}>
                  <ToggleField label="Profile Pic" name="profile_pic_present"
                               value={profile.profile_pic_present} onChange={handleChange} />
                  <ToggleField label="URL in Bio" name="url_in_bio"
                               value={profile.url_in_bio} onChange={handleChange} />
                </div>
                <div style={{ flex: 1 }}>
                  <ToggleField label="Verified" name="verified"
                               value={profile.verified} onChange={handleChange} />
                </div>
              </div>
            </div>

          </div>
        )}

        {/* ══════════════════ RESULT TAB ══════════════════ */}
        {tab === "result" && result && (
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20 }}>

            {/* Left: gauge + metrics */}
            <div>
              <div style={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 12,
                            padding: 24, textAlign: "center", marginBottom: 16 }}>
                <BCSGauge score={result.bcs_score} />
                <div style={{ display: "flex", gap: 10, marginTop: 20 }}>
                  <MetricCard label="PREDICTION"
                              value={result.is_fake ? "FAKE" : "GENUINE"}
                              color={result.is_fake ? C.red : C.green} />
                  <MetricCard label="FAKE PROB"
                              value={`${(result.probability * 100).toFixed(1)}%`}
                              color={result.probability > 0.6 ? C.red : result.probability > 0.35 ? C.yellow : C.green} />
                </div>
              </div>
            </div>

            {/* Right: radar + sub-scores */}
            <div style={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 12, padding: 20 }}>
              <div style={{ color: C.accentLt, fontSize: 11, marginBottom: 12, letterSpacing: "0.1em" }}>
                🕸 BEHAVIORAL SUB-SCORES
              </div>
              <div style={{ display: "flex", justifyContent: "center", marginBottom: 16 }}>
                <RadarChart scores={result} />
              </div>
              <SubScoreBar label="Posting Regularity"      value={result.posting_regularity} />
              <SubScoreBar label="Engagement Authenticity" value={result.engagement_authenticity} />
              <SubScoreBar label="Profile Completeness"    value={result.profile_completeness} />
              <SubScoreBar label="Content Quality"         value={result.content_quality} />
              <SubScoreBar label="Anti-Spam Signal"        value={result.spam_signal} />
            </div>

          </div>
        )}

        {/* ══════════════════ ABOUT TAB ══════════════════ */}
        {tab === "about" && (
          <div style={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 12, padding: 28 }}>
            <div style={{ color: C.accentLt, fontSize: 13, fontWeight: 700, marginBottom: 8 }}>
              ML-Based Fake Profile Detection Using Behavioral Consistency Score
            </div>
            <div style={{ color: C.muted, fontSize: 10, marginBottom: 24, letterSpacing: "0.08em" }}>
              FINAL YEAR PROJECT
            </div>
            {[
              { title: "System Overview",
                text: "This system computes a Behavioral Consistency Score (BCS) for social media profiles by analysing posting patterns, engagement, profile completeness, content quality, and spam signals. A Random Forest classifier trained on these features determines whether a profile is genuine or fake." },
              { title: "BCS Formula",
                text: "BCS = 0.20×PostingRegularity + 0.25×EngagementAuthenticity + 0.20×ProfileCompleteness + 0.20×ContentQuality + 0.15×AntiSpamSignal. Scores ≥70 → Genuine, 45–70 → Suspicious, <45 → Likely Fake." },
              { title: "Technology Stack",
                text: "Python · scikit-learn · Random Forest · Gradient Boosting · Flask REST API · React (frontend) · pandas · NumPy · Matplotlib / Seaborn" },
              { title: "Models Evaluated",
                text: "Random Forest · Gradient Boosting · Logistic Regression · SVM — evaluated using Accuracy, Precision, Recall, F1-Score, and ROC-AUC with 5-fold cross-validation." },
            ].map(({ title, text }) => (
              <div key={title} style={{ marginBottom: 20 }}>
                <div style={{ color: C.text, fontWeight: 700, fontSize: 11, marginBottom: 6 }}>{title}</div>
                <div style={{ color: C.muted, fontSize: 11, lineHeight: 1.7 }}>{text}</div>
              </div>
            ))}
          </div>
        )}

        {/* ── Quick-switch hint ── */}
        <div style={{ marginTop: 16, textAlign: "center", color: C.muted, fontSize: 10 }}>
          Adjust sliders in INPUT tab → see live analysis in RESULT tab
        </div>

      </div>
    </div>
  );
}
