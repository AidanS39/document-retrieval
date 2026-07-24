import { useState, useEffect, useCallback, useRef } from 'react';
import type { Query, PageInfo, Score, Scores, QueryCompletion } from './types';
import { fetchQueries, fetchPages, fetchAnnotations, fetchStatus, submitAnnotation, clearAnnotation, fetchNote, submitNote } from './api';
import Header from './components/Header';
import LeftPanel from './components/LeftPanel';
import PDFViewer from './components/PDFViewer';
import Footer from './components/Footer';

export default function App() {
  const [annotator, setAnnotatorRaw] = useState<string>(
    () => localStorage.getItem('annotator') ?? ''
  );
  const [showReg, setShowReg] = useState(() => !localStorage.getItem('annotator')?.trim());
  const [regName, setRegName] = useState('');
  const regInputRef = useRef<HTMLInputElement>(null);
  const [queries, setQueries] = useState<Query[]>([]);
  const [queryIndex, setQueryIndex] = useState(0);
  const [pages, setPages] = useState<PageInfo[]>([]);
  const [scores, setScores] = useState<Scores>({});
  const [focusedPageIdx, setFocusedPageIdx] = useState(0);
  const [queryCompletion, setQueryCompletion] = useState<QueryCompletion>({});
  const [loading, setLoading] = useState(true);
  const [poolMissing, setPoolMissing] = useState(false);
  const [note, setNote] = useState('');
  const noteSaveTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const setAnnotator = (name: string) => {
    setAnnotatorRaw(name);
    localStorage.setItem('annotator', name);
  };

  const submitReg = () => {
    const name = regName.trim();
    if (!name) return;
    setAnnotator(name);
    setShowReg(false);
  };

  useEffect(() => {
    if (showReg) regInputRef.current?.focus();
  }, [showReg]);

  // Load query list once on mount
  useEffect(() => {
    fetchQueries()
      .then(q => { setQueries(q); setLoading(false); })
      .catch(() => { setPoolMissing(true); setLoading(false); });
  }, []);

  // Reload per-annotator completion status when annotator name changes
  useEffect(() => {
    if (!annotator.trim()) { setQueryCompletion({}); return; }
    fetchStatus(annotator).then(setQueryCompletion);
  }, [annotator]);

  const currentQuery = queries[queryIndex] ?? null;

  // Load pages when query changes
  useEffect(() => {
    if (!currentQuery) return;
    setPages([]);
    setFocusedPageIdx(0);
    fetchPages(currentQuery.id).then(setPages);
  }, [currentQuery?.id]);

  // Load existing scores when query or annotator changes
  useEffect(() => {
    if (!currentQuery || !annotator.trim()) { setScores({}); return; }
    fetchAnnotations(annotator, currentQuery.id).then(existing => {
      setScores(existing as Scores);
      // Sync completion count from loaded data
      setQueryCompletion(prev => ({
        ...prev,
        [currentQuery.id]: Object.keys(existing).length,
      }));
    });
  }, [currentQuery?.id, annotator]);

  // Load note when query or annotator changes
  useEffect(() => {
    if (noteSaveTimer.current) { clearTimeout(noteSaveTimer.current); noteSaveTimer.current = null; }
    if (!currentQuery || !annotator.trim()) { setNote(''); return; }
    fetchNote(annotator, currentQuery.id).then(({ note: n }) => setNote(n));
  }, [currentQuery?.id, annotator]);

  const handleNoteChange = useCallback((text: string) => {
    setNote(text);
    if (!currentQuery || !annotator.trim()) return;
    if (noteSaveTimer.current) clearTimeout(noteSaveTimer.current);
    noteSaveTimer.current = setTimeout(() => {
      submitNote(annotator, currentQuery.id, text);
    }, 600);
  }, [annotator, currentQuery]);

  const focusedPage = pages[focusedPageIdx] ?? null;

  const score = useCallback((s: Score) => {
    if (!focusedPage || !annotator.trim() || !currentQuery) return;
    const pageId = focusedPage.page_id;

    setScores(prev => {
      const next = { ...prev, [pageId]: s };
      setQueryCompletion(qc => ({
        ...qc,
        [currentQuery.id]: Object.keys(next).length,
      }));
      return next;
    });

    submitAnnotation(annotator, currentQuery.id, pageId, s);

    // Advance focus to next unscored page, or just the next page
    setFocusedPageIdx(idx => {
      const nextUnscored = pages.findIndex(
        (p, i) => i > idx && scores[p.page_id] === undefined && p.page_id !== pageId
      );
      if (nextUnscored !== -1) return nextUnscored;
      return Math.min(idx + 1, pages.length - 1);
    });
  }, [focusedPage, annotator, currentQuery, pages, scores]);

  const clear = useCallback((pageId: number) => {
    if (!annotator.trim() || !currentQuery) return;
    setScores(prev => {
      const next = { ...prev };
      delete next[pageId];
      setQueryCompletion(qc => ({
        ...qc,
        [currentQuery.id]: Object.keys(next).length,
      }));
      return next;
    });
    clearAnnotation(annotator, currentQuery.id, pageId);
  }, [annotator, currentQuery]);

  const goToQuery = useCallback((idx: number) => {
    setQueryIndex(Math.max(0, Math.min(idx, queries.length - 1)));
  }, [queries.length]);

  // Global keyboard handler
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement).tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA') return;
      if (e.key >= '0' && e.key <= '3') {
        const s = parseInt(e.key) as Score;
        if (focusedPage && scores[focusedPage.page_id] === s) clear(focusedPage.page_id);
        else score(s);
      } else if (e.key === 'ArrowDown' || e.key === 'j') {
        setFocusedPageIdx(i => Math.min(i + 1, pages.length - 1));
      } else if (e.key === 'ArrowUp' || e.key === 'k') {
        setFocusedPageIdx(i => Math.max(i - 1, 0));
      } else if (e.key === 'ArrowRight' || e.key === 'l') {
        goToQuery(queryIndex + 1);
      } else if (e.key === 'ArrowLeft' || e.key === 'h') {
        goToQuery(queryIndex - 1);
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [score, clear, focusedPage, scores, pages.length, queryIndex, goToQuery]);

  if (loading) return <div className="loading">Loading…</div>;

  if (poolMissing) {
    return (
      <div className="no-pool">
        <strong>Query pool not found</strong>
        <span>Run <code>setup_evaluation.py</code> to generate it first.</span>
      </div>
    );
  }

  const queryComplete = currentQuery
    ? (queryCompletion[currentQuery.id] ?? 0) >= currentQuery.page_count
    : false;

  return (
    <div className="app">
      {showReg && (
        <div className="reg-backdrop">
          <div className="reg-modal">
            <div className="reg-modal-title">Welcome to NSTX Annotator</div>
            <p className="reg-modal-desc">Enter your name to begin annotating.</p>
            <input
              ref={regInputRef}
              className="reg-input"
              type="text"
              placeholder="Your name"
              value={regName}
              onChange={e => setRegName(e.target.value)}
              onKeyDown={e => { if (e.key === 'Enter') submitReg(); }}
            />
            <button className="reg-submit-btn" onClick={submitReg} disabled={!regName.trim()}>
              Start annotating
            </button>
          </div>
        </div>
      )}
      <Header
        annotator={annotator}
        onAnnotatorChange={setAnnotator}
        queryIndex={queryIndex}
        totalQueries={queries.length}
        queryComplete={queryComplete}
      />
      <div className="main-layout">
        <LeftPanel
          query={currentQuery?.query ?? ''}
          pages={pages}
          scores={scores}
          focusedPageIdx={focusedPageIdx}
          onFocus={setFocusedPageIdx}
          onScore={(pageId, s) => {
            if (!annotator.trim() || !currentQuery) return;
            setScores(prev => {
              const next = { ...prev, [pageId]: s };
              setQueryCompletion(qc => ({
                ...qc,
                [currentQuery.id]: Object.keys(next).length,
              }));
              return next;
            });
            submitAnnotation(annotator, currentQuery.id, pageId, s);
          }}
          onClear={clear}
          note={note}
          onNoteChange={handleNoteChange}
        />
        <PDFViewer
          pdfUrl={focusedPage?.pdf_url ?? null}
          pageNumber={focusedPage?.page_number ?? 1}
        />
      </div>
      <Footer
        queryIndex={queryIndex}
        totalQueries={queries.length}
        queries={queries}
        queryCompletion={queryCompletion}
        onPrev={() => goToQuery(queryIndex - 1)}
        onNext={() => goToQuery(queryIndex + 1)}
        onGoTo={goToQuery}
      />
    </div>
  );
}
