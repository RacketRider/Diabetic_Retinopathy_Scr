import { useState, useEffect } from 'react';
import { checkHealth } from '@/api/client';

export function Header() {
  const [backendStatus, setBackendStatus] = useState<'connected' | 'disconnected' | 'checking'>('checking');

  useEffect(() => {
    const check = async () => {
      try {
        await checkHealth();
        setBackendStatus('connected');
      } catch {
        setBackendStatus('disconnected');
      }
    };
    check();
    const interval = setInterval(check, 30_000);
    return () => clearInterval(interval);
  }, []);

  return (
    <header className="bg-white border-b border-gray-200 sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-14">
          <div className="flex items-center gap-2">
            <span className="text-xl">🔬</span>
            <h1 className="text-lg font-bold text-gray-900">DR-Screen AI</h1>
            <span className="hidden sm:inline text-xs text-gray-400 ml-2">SIH26038</span>
          </div>

          <div className="flex items-center gap-3">
            <div className="flex items-center gap-1.5 text-xs">
              <span
                className={`w-2 h-2 rounded-full ${
                  backendStatus === 'connected'
                    ? 'bg-green-500'
                    : backendStatus === 'disconnected'
                    ? 'bg-red-400'
                    : 'bg-yellow-400 animate-pulse'
                }`}
              />
              <span className="text-gray-500">
                {backendStatus === 'connected'
                  ? 'Backend connected'
                  : backendStatus === 'disconnected'
                  ? 'Backend offline'
                  : 'Checking...'}
              </span>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
}
