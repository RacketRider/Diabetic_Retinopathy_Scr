import { useState } from 'react';
import { useImageUpload } from '@/hooks/useImageUpload';
import { usePrediction } from '@/hooks/usePrediction';
import { Header } from '@/components/Layout/Header';
import { Footer } from '@/components/Layout/Footer';
import { MedicalDisclaimer } from '@/components/common/MedicalDisclaimer';
import { Card } from '@/components/common/Card';
import { LoadingSpinner } from '@/components/common/LoadingSpinner';
import { DropZone } from '@/components/Upload/DropZone';
import { ImagePreview } from '@/components/Upload/ImagePreview';
import { GradingCard } from '@/components/Results/GradingCard';
import { ConfidenceBar } from '@/components/Results/ConfidenceBar';
import { GradCAMOverlay } from '@/components/Results/GradCAMOverlay';
import { LesionEvidence } from '@/components/Results/LesionEvidence';
import { ClinicalReport } from '@/components/Report/ClinicalReport';
import { SimulinkViz } from '@/components/Dashboard/SimulinkViz';
import { StatsPanel } from '@/components/Dashboard/StatsPanel';
import { formatMs } from '@/utils/formatters';
import type { GradCAMResponse } from '@/types';

type Tab = 'screening' | 'pipeline' | 'stats';

export default function App() {
  const upload = useImageUpload();
  const prediction = usePrediction();
  const [activeTab, setActiveTab] = useState<Tab>('screening');
  const [scanHistory, setScanHistory] = useState<GradCAMResponse[]>([]);

  const handleAnalyze = () => {
    if (upload.file) {
      prediction.analyze(upload.file);
    }
  };

  const handleNewScan = () => {
    // Save completed result to session history before clearing
    if (prediction.result) {
      setScanHistory((prev) => [...prev, prediction.result!]);
    }
    upload.clear();
    prediction.reset();
  };

  const showResults = prediction.result && !prediction.isLoading;

  const tabs: { key: Tab; label: string; icon: string }[] = [
    { key: 'screening', label: 'Screening', icon: '🔬' },
    { key: 'pipeline', label: 'Pipeline', icon: '📡' },
    { key: 'stats', label: 'Statistics', icon: '📊' },
  ];

  return (
    <div className="min-h-screen flex flex-col">
      <Header />

      <main className="flex-1 max-w-5xl mx-auto w-full px-4 sm:px-6 lg:px-8 py-6 space-y-6">
        <MedicalDisclaimer />

        {/* Navigation Tabs */}
        <div className="flex border-b border-gray-200">
          {tabs.map((tab) => (
            <button
              key={tab.key}
              onClick={() => setActiveTab(tab.key)}
              className={`flex items-center gap-1.5 px-4 py-2.5 text-sm font-medium border-b-2 transition-colors ${
                activeTab === tab.key
                  ? 'border-primary-500 text-primary-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              <span>{tab.icon}</span>
              {tab.label}
              {tab.key === 'stats' && scanHistory.length > 0 && (
                <span className="ml-1 bg-primary-100 text-primary-600 text-xs font-bold px-1.5 py-0.5 rounded-full">
                  {scanHistory.length}
                </span>
              )}
            </button>
          ))}
        </div>

        {/* === Screening Tab === */}
        {activeTab === 'screening' && (
          <>
            {/* Upload View */}
            {!showResults && !prediction.isLoading && (
              <Card title="Fundus Image Upload">
                {!upload.preview ? (
                  <DropZone
                    onFile={upload.processFile}
                    isValidating={upload.isValidating}
                    error={upload.error}
                  />
                ) : (
                  <ImagePreview
                    src={upload.preview}
                    onClear={handleNewScan}
                    onAnalyze={handleAnalyze}
                    isLoading={prediction.isLoading}
                  />
                )}

                {prediction.error && (
                  <div className="mt-4 bg-red-50 border border-red-200 rounded-lg px-4 py-2.5 text-sm text-red-700">
                    ❌ {prediction.error}
                  </div>
                )}
              </Card>
            )}

            {/* Loading View */}
            {prediction.isLoading && <LoadingSpinner message="Running DR analysis..." />}

            {/* Results View */}
            {showResults && prediction.result && (
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <h2 className="text-lg font-bold text-gray-900">Screening Results</h2>
                  <div className="flex items-center gap-3">
                    <span className="text-xs text-gray-400">
                      ⏱ {formatMs(prediction.result.processing_time_ms)}
                    </span>
                    <button
                      onClick={handleNewScan}
                      className="px-3 py-1.5 text-sm text-primary-600 hover:bg-primary-50 rounded-lg transition-colors"
                    >
                      ← New Scan
                    </button>
                  </div>
                </div>

                {/* Grading */}
                <GradingCard
                  grading={prediction.result.grading}
                  confidence={prediction.result.confidence}
                />

                {/* Grad-CAM + Confidence side by side */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                  <Card title="Grad-CAM Explainability">
                    {upload.preview && prediction.result.gradcam && (
                      <GradCAMOverlay
                        originalSrc={upload.preview}
                        overlayBase64={prediction.result.gradcam.overlay_base64}
                      />
                    )}
                  </Card>

                  <Card title="Class Probabilities">
                    <ConfidenceBar
                      confidence={prediction.result.confidence}
                      predictedClass={prediction.result.grading.label}
                    />
                  </Card>
                </div>

                {/* Lesion Evidence */}
                {prediction.result.gradcam && (
                  <Card title="Lesion Evidence">
                    <LesionEvidence
                      regions={prediction.result.gradcam.attention_regions}
                      clinicalEvidence={prediction.result.gradcam.clinical_evidence}
                    />
                  </Card>
                )}

                {/* Report Download */}
                {upload.file && (
                  <Card title="Clinical Report">
                    <ClinicalReport file={upload.file} />
                  </Card>
                )}
              </div>
            )}
          </>
        )}

        {/* === Pipeline Tab === */}
        {activeTab === 'pipeline' && <SimulinkViz />}

        {/* === Statistics Tab === */}
        {activeTab === 'stats' && (
          <>
            <StatsPanel scans={scanHistory} />
            {scanHistory.length === 0 && (
              <div className="text-center py-12 text-gray-400">
                <div className="text-4xl mb-3">📊</div>
                <p className="text-sm">No scans completed yet. Run a screening to see statistics.</p>
              </div>
            )}
          </>
        )}

        {/* Privacy notice */}
        <div className="text-center text-xs text-gray-400 py-4">
          🔒 Your image is processed in-memory and never stored to disk.
        </div>
      </main>

      <Footer />
    </div>
  );
}
