export interface Query {
  id: number;
  query: string;
  page_count: number;
}

export interface PageInfo {
  page_id: number;
  page_number: number;
  document_name: string;
  image_url: string;
  pdf_url: string;
}

export type Score = 0 | 1 | 2 | 3;
export type Scores = Record<number, Score>;
// query_id -> number of annotated pages
export type QueryCompletion = Record<number, number>;
