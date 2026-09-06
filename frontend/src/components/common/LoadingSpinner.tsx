export function LoadingSpinner({ message = 'Analyzing...' }: { message?: string }) {
  return (
    <div className="flex flex-col items-center justify-center py-20 gap-4">
      <div className="relative w-16 h-16">
        <div className="absolute inset-0 border-4 border-primary-200 rounded-full" />
        <div className="absolute inset-0 border-4 border-primary-600 rounded-full border-t-transparent animate-spin" />
      </div>
      <p className="text-gray-600 text-sm font-medium">{message}</p>
      <p className="text-gray-400 text-xs">Image is processed in-memory only — never stored</p>
    </div>
  );
}
