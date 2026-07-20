import { useRef, useEffect } from 'react';
import type { PageInfo, Score, Scores } from '../types';

const SCORE_LABELS: Score[] = [0, 1, 2, 3];

interface PageItemProps {
  page: PageInfo;
  scores: Scores;
  focused: boolean;
  onFocus: () => void;
  onScore: (pageId: number, score: Score) => void;
  onClear: (pageId: number) => void;
}

export default function PageItem({ page, scores, focused, onFocus, onScore, onClear }: PageItemProps) {
  const ref = useRef<HTMLDivElement>(null);
  const currentScore = scores[page.page_id];
  const scored = currentScore !== undefined;

  useEffect(() => {
    if (focused) {
      ref.current?.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
    }
  }, [focused]);

  return (
    <div
      ref={ref}
      className={`page-item${focused ? ' focused' : ''}`}
      onClick={onFocus}
    >
      <div className="page-item-header">
        {scored
          ? <span className="page-checkmark">✓</span>
          : <span className="page-checkmark-placeholder" />
        }
        <span className="page-label">
          <strong>p.{page.page_number}</strong> — {page.document_name}
        </span>
      </div>

      <div className="score-buttons">
        {SCORE_LABELS.map(s => (
          <button
            key={s}
            className={`score-btn${currentScore === s ? ` active-${s}` : ''}`}
            onClick={e => { e.stopPropagation(); currentScore === s ? onClear(page.page_id) : onScore(page.page_id, s); }}
          >
            {s}
          </button>
        ))}
      </div>

      {focused && (
        <div className="page-meta">
          Document: <span>{page.document_name}</span><br />
          Page: <span>{page.page_number}</span><br />
          Score: <span>{scored ? currentScore : 'unscored'}</span><br />
          <span style={{ color: '#555', fontSize: '11px' }}>Keys 0–3 to score · ↑↓ to navigate</span>
        </div>
      )}
    </div>
  );
}
