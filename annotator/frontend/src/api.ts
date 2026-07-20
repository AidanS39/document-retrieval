import type { Query, PageInfo, Score, Scores, QueryCompletion } from './types';

async function request<T>(url: string, options?: RequestInit): Promise<T> {
  const res = await fetch(url, options);
  if (!res.ok) throw new Error(`${options?.method ?? 'GET'} ${url} failed: ${res.status}`);
  if (res.status === 204) return undefined as T;
  return res.json();
}

export const fetchQueries = (): Promise<Query[]> =>
  request('/api/queries');

export const fetchPages = (queryId: number): Promise<PageInfo[]> =>
  request(`/api/queries/${queryId}/pages`);

export const fetchAnnotations = (annotator: string, queryId: number): Promise<Scores> =>
  request(`/api/annotations/${encodeURIComponent(annotator)}/${queryId}`);

export const fetchStatus = (annotator: string): Promise<QueryCompletion> =>
  request(`/api/status/${encodeURIComponent(annotator)}`);

export const submitAnnotation = (
  annotator: string,
  queryId: number,
  pageId: number,
  score: Score,
): Promise<void> =>
  request('/api/annotations', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ annotator, query_id: queryId, page_id: pageId, score }),
  });

export const clearAnnotation = (
  annotator: string,
  queryId: number,
  pageId: number,
): Promise<void> =>
  request(`/api/annotations/${encodeURIComponent(annotator)}/${queryId}/${pageId}`, {
    method: 'DELETE',
  });
