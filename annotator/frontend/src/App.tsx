import { useState, useEffect, useCallback, useRef } from 'react';
import type { Query, PageInfo, Score, Scores, QueryCompletion } from './types';
import { fetchQueries, fetchPages, fetchAnnotations, fetchStatus, submitAnnotation, clearAnnotation, fetchNote, submitNote, fetchAnnotatorByName, createAnnotator } from './api';
import Header from './components/Header';
import LeftPanel from './components/LeftPanel';
import PDFViewer from './components/PDFViewer';
import Footer from './components/Footer';

export default function App() {
  const [annotatorId, setAnnotatorId] = useState<number | null>(() => {
    const stored = localStorage.getItem('annotatorId');
    return stored ? parseInt(stored, 10) : null;
  });
  const [annotatorName, setAnnotatorName] = useState<string>(
    () => localStorage.getItem('annotatorName') ?? ''
  );
  const [showSignIn, setShowSignIn] = useState(() => !localStorage.getItem('annotatorId'));
  const [showCreate, setShowCreate] = useState(false);
  const [signInName, setSignInName] = useState('');
  const [signInError, setSignInError] = useState('');
  const [createName, setCreateName] = useState('');
  const [createError, setCreateError] = useState('');
  const signInInputRef = useRef<HTMLInputElement>(null);
  const createInputRef = useRef<HTMLInputElement>(null);
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

  useEffect(() => {
    if (showSignIn && !showCreate) signInInputRef.current?.focus();
  }, [showSignIn, showCreate]);

  useEffect(() => {
    if (showCreate) createInputRef.current?.focus();
  }, [showCreate]);

  const signIn = async () => {
    const name = signInName.trim();
    if (!name) return;
    try {
      const info = await fetchAnnotatorByName(name);
      localStorage.setItem('annotatorId', String(info.id));
      localStorage.setItem('annotatorName', info.name);
      setAnnotatorId(info.id);
      setAnnotatorName(info.name);
      setShowSignIn(false);
      setSignInError('');
      setSignInName('');
    } catch {
      setSignInError(`No annotator found with name "${name}".`);
    }
  };

  const openCreate = () => {
    setCreateName(signInName.trim());
    setCreateError('');
    setShowCreate(true);
  };

  const createAccount = async () => {
    const name = createName.trim();
    if (!name) return;
    try {
      const info = await createAnnotator(name);
      localStorage.setItem('annotatorId', String(info.id));
      localStorage.setItem('annotatorName', info.name);
      setAnnotatorId(info.id);
      setAnnotatorName(info.name);
      setShowCreate(false);
      setShowSignIn(false);
      setCreateError('');
      setCreateName('');
    } catch {
      setCreateError(`An annotator named "${createName.trim()}" already exists.`);
    }
  };

  const signOut = () => {
    localStorage.removeItem('annotatorId');
    localStorage.removeItem('annotatorName');
    setAnnotatorId(null);
    setAnnotatorName('');
    setShowSignIn(true);
    setShowCreate(false);
    setScores({});
    setQueryCompletion({});
    setNote('');
    setSignInName('');
    setSignInError('');
  };

  // Load query list once on mount
  useEffect(() => {
    fetchQueries()
      .then(q => { setQueries(q); setLoading(false); })
      .catch(() => { setPoolMissing(true); setLoading(false); });
  }, []);

  // Reload per-annotator completion status when annotator changes
  useEffect(() => {
    if (!annotatorId) { setQueryCompletion({}); return; }
    fetchStatus(annotatorId).then(setQueryCompletion);
  }, [annotatorId]);

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
    if (!currentQuery || !annotatorId) { setScores({}); return; }
    fetchAnnotations(annotatorId, currentQuery.id).then(existing => {
      setScores(existing as Scores);
      setQueryCompletion(prev => ({
        ...prev,
        [currentQuery.id]: Object.keys(existing).length,
      }));
    });
  }, [currentQuery?.id, annotatorId]);

  // Load note when query or annotator changes
  useEffect(() => {
    if (noteSaveTimer.current) { clearTimeout(noteSaveTimer.current); noteSaveTimer.current = null; }
    if (!currentQuery || !annotatorName) { setNote(''); return; }
    fetchNote(annotatorName, currentQuery.id).then(({ note: n }) => setNote(n));
  }, [currentQuery?.id, annotatorName]);

  const handleNoteChange = useCallback((text: string) => {
    setNote(text);
    if (!currentQuery || !annotatorName) return;
    if (noteSaveTimer.current) clearTimeout(noteSaveTimer.current);
    noteSaveTimer.current = setTimeout(() => {
      submitNote(annotatorName, currentQuery.id, text);
    }, 600);
  }, [annotatorName, currentQuery]);

  const focusedPage = pages[focusedPageIdx] ?? null;

  const score = useCallback((s: Score) => {
    if (!focusedPage || !annotatorId || !currentQuery) return;
    const pageId = focusedPage.page_id;

    setScores(prev => {
      const next = { ...prev, [pageId]: s };
      setQueryCompletion(qc => ({
        ...qc,
        [currentQuery.id]: Object.keys(next).length,
      }));
      return next;
    });

    submitAnnotation(annotatorId, currentQuery.id, pageId, s);

    setFocusedPageIdx(idx => {
      const nextUnscored = pages.findIndex(
        (p, i) => i > idx && scores[p.page_id] === undefined && p.page_id !== pageId
      );
      if (nextUnscored !== -1) return nextUnscored;
      return Math.min(idx + 1, pages.length - 1);
    });
  }, [focusedPage, annotatorId, currentQuery, pages, scores]);

  const clear = useCallback((pageId: number) => {
    if (!annotatorId || !currentQuery) return;
    setScores(prev => {
      const next = { ...prev };
      delete next[pageId];
      setQueryCompletion(qc => ({
        ...qc,
        [currentQuery.id]: Object.keys(next).length,
      }));
      return next;
    });
    clearAnnotation(annotatorId, currentQuery.id, pageId);
  }, [annotatorId, currentQuery]);

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
      {showSignIn && !showCreate && (
        <div className="reg-backdrop">
          <div className="reg-modal">
            <div className="reg-modal-title">Sign in</div>
            <p className="reg-modal-desc">Enter your name to continue annotating.</p>
            <input
              ref={signInInputRef}
              className="reg-input"
              type="text"
              placeholder="Your name"
              value={signInName}
              onChange={e => { setSignInName(e.target.value); setSignInError(''); }}
              onKeyDown={e => { if (e.key === 'Enter') signIn(); }}
            />
            {signInError && <p className="auth-error">{signInError}</p>}
            <button className="reg-submit-btn" onClick={signIn} disabled={!signInName.trim()}>
              Sign in
            </button>
            <button className="auth-link-btn" onClick={openCreate}>
              Create an account
            </button>
          </div>
        </div>
      )}
      {showCreate && (
        <div className="reg-backdrop">
          <div className="reg-modal">
            <div className="reg-modal-title">Create account</div>
            <p className="reg-modal-desc">Choose a name to register as a new annotator.</p>
            <input
              ref={createInputRef}
              className="reg-input"
              type="text"
              placeholder="Your name"
              value={createName}
              onChange={e => { setCreateName(e.target.value); setCreateError(''); }}
              onKeyDown={e => { if (e.key === 'Enter') createAccount(); }}
            />
            {createError && <p className="auth-error">{createError}</p>}
            <button className="reg-submit-btn" onClick={createAccount} disabled={!createName.trim()}>
              Create account
            </button>
            <button className="auth-link-btn" onClick={() => { setShowCreate(false); setCreateError(''); }}>
              Back to sign in
            </button>
          </div>
        </div>
      )}
      <Header
        annotator={annotatorName}
        onSignOut={signOut}
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
            if (!annotatorId || !currentQuery) return;
            setScores(prev => {
              const next = { ...prev, [pageId]: s };
              setQueryCompletion(qc => ({
                ...qc,
                [currentQuery.id]: Object.keys(next).length,
              }));
              return next;
            });
            submitAnnotation(annotatorId, currentQuery.id, pageId, s);
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
