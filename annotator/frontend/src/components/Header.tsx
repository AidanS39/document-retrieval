import { useState, useEffect, useRef } from 'react';

const SCORE_DESCRIPTIONS = [
  { score: 0, label: 'Not relevant', color: 'var(--score-0)', description: 'The page contains no useful information for this query.' },
  { score: 1, label: 'Marginally relevant', color: 'var(--score-1)', description: 'The page touches on the topic but does not directly address the query.' },
  { score: 2, label: 'Relevant', color: 'var(--score-2)', description: 'The page contains information that directly relates to the query.' },
  { score: 3, label: 'Highly relevant', color: 'var(--score-3)', description: 'The page substantially and directly addresses the query.' },
];

interface HeaderProps {
  annotator: string;
  onAnnotatorChange: (name: string) => void;
  queryIndex: number;
  totalQueries: number;
  queryComplete: boolean;
}

export default function Header({
  annotator,
  onAnnotatorChange,
  queryIndex,
  totalQueries,
  queryComplete,
}: HeaderProps) {
  const [showInfo, setShowInfo] = useState(false);
  const modalRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!showInfo) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') setShowInfo(false); };
    const onClick = (e: MouseEvent) => {
      if (modalRef.current && !modalRef.current.contains(e.target as Node)) setShowInfo(false);
    };
    window.addEventListener('keydown', onKey);
    window.addEventListener('mousedown', onClick);
    return () => {
      window.removeEventListener('keydown', onKey);
      window.removeEventListener('mousedown', onClick);
    };
  }, [showInfo]);

  return (
    <>
      <header className="header">
        <span className="header-title">NSTX Annotator</span>
        <input
          className="annotator-input"
          type="text"
          placeholder="Your name…"
          value={annotator}
          onChange={e => onAnnotatorChange(e.target.value)}
        />
        <button className="info-btn" onClick={() => setShowInfo(true)} title="Scoring guide">
          ?
        </button>
        <div className="header-spacer" />
        {totalQueries > 0 && (
          <span className="query-progress">
            Query {queryIndex + 1} / {totalQueries}
            {queryComplete && <span className="query-complete-badge">✓</span>}
          </span>
        )}
      </header>

      {showInfo && (
        <div className="info-backdrop">
          <div className="info-modal" ref={modalRef}>
            <div className="info-modal-header">
              <span>Scoring Guide</span>
              <button className="info-close-btn" onClick={() => setShowInfo(false)}>✕</button>
            </div>
            <div className="info-modal-body">
              {SCORE_DESCRIPTIONS.map(({ score, label, color, description }) => (
                <div key={score} className="info-score-row">
                  <span className="info-score-badge" style={{ background: color }}>{score}</span>
                  <div>
                    <div className="info-score-label">{label}</div>
                    <div className="info-score-desc">{description}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </>
  );
}
