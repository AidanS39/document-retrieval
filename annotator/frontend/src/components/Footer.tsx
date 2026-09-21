import type { Query, QueryCompletion } from '../types';

interface FooterProps {
  queryIndex: number;
  totalQueries: number;
  queries: Query[];
  queryCompletion: QueryCompletion;
  onPrev: () => void;
  onNext: () => void;
  onGoTo: (index: number) => void;
}

export default function Footer({
  queryIndex,
  totalQueries,
  queries,
  queryCompletion,
  onPrev,
  onNext,
  onGoTo,
}: FooterProps) {
  return (
    <footer className="footer">
      <button className="nav-btn" onClick={onPrev} disabled={queryIndex === 0}>
        ← Previous
      </button>

      <div className="query-dots">
        {queries.map((q, i) => {
          const complete = (queryCompletion[q.id] ?? 0) >= q.page_count;
          const current = i === queryIndex;
          return (
            <button
              key={q.id}
              className={[
                'query-dot',
                current ? 'dot-current' : '',
                complete ? 'dot-complete' : '',
              ].join(' ')}
              title={`Query ${i + 1}: ${q.query.slice(0, 60)}${complete ? ' ✓' : ''}`}
              onClick={() => onGoTo(i)}
            />
          );
        })}
      </div>

      <button className="nav-btn" onClick={onNext} disabled={queryIndex >= totalQueries - 1}>
        Next →
      </button>
    </footer>
  );
}
