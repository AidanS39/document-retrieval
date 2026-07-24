import type { PageInfo, Score, Scores } from '../types';
import PageItem from './PageItem';

interface LeftPanelProps {
  query: string;
  pages: PageInfo[];
  scores: Scores;
  focusedPageIdx: number;
  onFocus: (idx: number) => void;
  onScore: (pageId: number, score: Score) => void;
  onClear: (pageId: number) => void;
  note: string;
  onNoteChange: (text: string) => void;
}

export default function LeftPanel({
  query,
  pages,
  scores,
  focusedPageIdx,
  onFocus,
  onScore,
  onClear,
  note,
  onNoteChange,
}: LeftPanelProps) {
  return (
    <div className="left-panel">
      <div className={`query-text-box${pages.length > 0 && pages.every(p => scores[p.page_id] !== undefined) ? ' query-text-box--complete' : ''}`}>
        <div className="query-text-box-header">
          <div className="query-label">Query</div>
          {pages.length > 0 && (
            <div className="query-progress-inline">
              {pages.filter(p => scores[p.page_id] !== undefined).length} / {pages.length} annotated
            </div>
          )}
        </div>
        <div className="query-text">{query}</div>
      </div>

      <div className="page-list">
        {pages.length === 0 ? (
          <div style={{ padding: '24px 16px', color: '#666', fontSize: '13px' }}>
            Loading pages…
          </div>
        ) : (
          pages.map((page, idx) => (
            <PageItem
              key={page.page_id}
              page={page}
              scores={scores}
              focused={idx === focusedPageIdx}
              onFocus={() => onFocus(idx)}
              onScore={onScore}
              onClear={onClear}
            />
          ))
        )}
      </div>

      <div className="missing-pages-box">
        <div className="missing-pages-label">Missing relevant pages</div>
        <textarea
          className="missing-pages-textarea"
          placeholder="Note any pages you know are relevant but weren't included…"
          value={note}
          onChange={e => onNoteChange(e.target.value)}
        />
      </div>
    </div>
  );
}
