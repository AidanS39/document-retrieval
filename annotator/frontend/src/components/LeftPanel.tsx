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
}

export default function LeftPanel({
  query,
  pages,
  scores,
  focusedPageIdx,
  onFocus,
  onScore,
  onClear,
}: LeftPanelProps) {
  return (
    <div className="left-panel">
      <div className="query-text-box">
        <div className="query-label">Query</div>
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
    </div>
  );
}
