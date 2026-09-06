export function Footer() {
  return (
    <footer className="bg-white border-t border-gray-200 mt-auto">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
        <div className="flex flex-col sm:flex-row items-center justify-between gap-2 text-xs text-gray-400">
          <p>SIH 2026 — Problem Statement SIH26038 — MathWorks</p>
          <p className="flex items-center gap-1">
            <span>🔒</span>
            Images processed in-memory only — zero disk persistence
          </p>
        </div>
      </div>
    </footer>
  );
}
