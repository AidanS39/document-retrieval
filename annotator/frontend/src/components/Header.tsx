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
        <button className="info-btn" onClick={() => setShowInfo(true)} title="Annotation guide">
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
              <span>Annotation Guide</span>
              <button className="info-close-btn" onClick={() => setShowInfo(false)}>✕</button>
            </div>
            <div className="info-modal-body">

              <div className="info-section">
                <div className="info-section-title">Purpose</div>
                <p className="info-section-text">
                  This tool collects human relevance judgements for NSTX document retrieval.
                  Your scores are used to build a ground-truth dataset that benchmarks how well
                  the retrieval system ranks pages against physicist queries.
                </p>
              </div>

              <div className="info-divider" />

              <div className="info-section">
                <div className="info-section-title">How to annotate</div>
                <ol className="info-steps">
                  <li>Enter your name in the annotator field in the header.</li>
                  <li>Read the query shown at the top of the left panel.</li>
                  <li>Click a page entry to open it in the PDF viewer on the right.</li>
                  <li>Score it using the 0–3 buttons or press the corresponding number key.</li>
                  <li>Move to the next page and repeat until all pages for the query are scored (a <strong>✓</strong> appears in the header).</li>
                  <li>Use the footer arrows or <kbd>←</kbd> / <kbd>→</kbd> to switch queries.</li>
                </ol>
                <p className="info-section-text">
                  Judge relevance <em>semantically</em> — a page doesn't need to use the exact words
                  of the query to be relevant. If the content provides information that can be used to
                  fully answer the question, treat it as relevant.
                </p>
                <p className="info-section-text">
                  Also consider pages that don't answer the query directly but provide background
                  context that would help a reader understand what the topic is. These are still
                  worth a score of 1 or 2 depending on how useful that context is.
                </p>
                <div className="info-shortcuts">
                  <span><kbd>↑</kbd><kbd>↓</kbd> or <kbd>j</kbd><kbd>k</kbd> — navigate pages</span>
                  <span><kbd>←</kbd><kbd>→</kbd> or <kbd>h</kbd><kbd>l</kbd> — navigate queries</span>
                  <span><kbd>0</kbd>–<kbd>3</kbd> — score focused page (re-press to clear)</span>
                </div>
              </div>

              <div className="info-divider" />

              <div className="info-section">
                <div className="info-section-title">Scoring guide</div>
                <div className="info-score-list">
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
          </div>
        </div>
      )}
    </>
  );
}
