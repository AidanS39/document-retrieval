import { Component, ReactNode, useState, useEffect, useRef } from 'react';
import { Document, Page } from 'react-pdf';
import 'react-pdf/dist/Page/AnnotationLayer.css';
import 'react-pdf/dist/Page/TextLayer.css';

// react-pdf destroys the worker transport in a cleanup effect, but a deferred
// Page mount effect may still fire against the null transport. An error boundary
// here catches that TypeError and resets when the URL settles.
class PDFErrorBoundary extends Component<
  { children: ReactNode; resetKey: string | null },
  { hasError: boolean }
> {
  state = { hasError: false };
  static getDerivedStateFromError() { return { hasError: true }; }
  componentDidUpdate(prevProps: { resetKey: string | null }) {
    if (prevProps.resetKey !== this.props.resetKey && this.state.hasError) {
      this.setState({ hasError: false });
    }
  }
  render() {
    if (this.state.hasError) {
      return <span style={{ color: '#666', padding: '32px' }}>Loading PDF…</span>;
    }
    return this.props.children;
  }
}

interface PDFViewerProps {
  pdfUrl: string | null;
  pageNumber: number;
}

export default function PDFViewer({ pdfUrl, pageNumber }: PDFViewerProps) {
  const [numPages, setNumPages] = useState<number>(0);
  const [debouncedUrl, setDebouncedUrl] = useState(pdfUrl);
  // Only set after onLoadSuccess — guarantees <Page> is never mounted mid-transition
  const [loadedUrl, setLoadedUrl] = useState<string | null>(null);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    if (timerRef.current) clearTimeout(timerRef.current);
    timerRef.current = setTimeout(() => setDebouncedUrl(pdfUrl), 150);
    return () => { if (timerRef.current) clearTimeout(timerRef.current); };
  }, [pdfUrl]);

  if (!pdfUrl) {
    return (
      <div className="pdf-panel">
        <div className="pdf-placeholder">Select a page to view its document</div>
      </div>
    );
  }

  const pageWidth = Math.min(window.innerWidth - 430, 900);
  // Guard pageNumber range: pdfUrl/pageNumber change immediately while debouncedUrl
  // still lags, so the new pageNumber may exceed the old document's count.
  const pageReady = numPages > 0 && loadedUrl === debouncedUrl
    && pageNumber >= 1 && pageNumber <= numPages;

  return (
    <div className="pdf-panel">
      {pageReady && (
        <div className="pdf-page-label">Page {pageNumber} / {numPages}</div>
      )}
      <div className="pdf-scroll">
        <PDFErrorBoundary resetKey={debouncedUrl}>
          <Document
            file={debouncedUrl}
            onLoadSuccess={({ numPages: n }) => {
              setNumPages(n);
              setLoadedUrl(debouncedUrl);
            }}
            loading={<span style={{ color: '#666', padding: '32px' }}>Loading PDF…</span>}
            error={<span style={{ color: '#e85c5c', padding: '32px' }}>Failed to load PDF</span>}
          >
            {pageReady && (
              <div className="pdf-page-wrapper">
                <Page
                  pageNumber={pageNumber}
                  width={pageWidth}
                  renderTextLayer={true}
                  renderAnnotationLayer={true}
                />
              </div>
            )}
          </Document>
        </PDFErrorBoundary>
      </div>
    </div>
  );
}
