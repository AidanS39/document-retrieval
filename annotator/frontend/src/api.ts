import type { AnnotatorInfo, Query, PageInfo, Score, Scores, QueryCompletion } from './types';

async function request<T>(url: string, options?: RequestInit): Promise<T> {
  const res = await fetch(url, options);
  if (!res.ok) throw new Error(`${options?.method ?? 'GET'} ${url} failed: ${res.status}`);
  if (res.status === 204) return undefined as T;
  return res.json();
}

export const fetchAnnotatorByName = (name: string): Promise<AnnotatorInfo> =>
  request(`/api/annotators/by-name/${encodeURIComponent(name)}`);

export const createAnnotator = (name: string): Promise<AnnotatorInfo> =>
  request('/api/annotators', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name }),
  });

export const fetchQueries = (): Promise<Query[]> =>
  request('/api/queries');

export const fetchPages = (queryId: number): Promise<PageInfo[]> =>
  request(`/api/queries/${queryId}/pages`);

export const fetchAnnotations = (annotatorId: number, queryId: number): Promise<Scores> =>
  request(`/api/annotations/${annotatorId}/${queryId}`);

export const fetchStatus = (annotatorId: number): Promise<QueryCompletion> =>
  request(`/api/status/${annotatorId}`);

export const submitAnnotation = (
  annotatorId: number,
  queryId: number,
  pageId: number,
  score: Score,
): Promise<void> =>
  request('/api/annotations', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ annotator_id: annotatorId, query_id: queryId, page_id: pageId, score }),
  });

export const clearAnnotation = (
  annotatorId: number,
  queryId: number,
  pageId: number,
): Promise<void> =>
  request(`/api/annotations/${annotatorId}/${queryId}/${pageId}`, {
    method: 'DELETE',
  });

export const fetchNote = (annotator: string, queryId: number): Promise<{ note: string }> =>
  request(`/api/notes/${encodeURIComponent(annotator)}/${queryId}`);

export const submitNote = (annotator: string, queryId: number, note: string): Promise<void> =>
  request('/api/notes', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ annotator, query_id: queryId, note }),
  });
