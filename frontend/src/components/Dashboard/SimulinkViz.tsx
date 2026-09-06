import { Card } from '@/components/common/Card';

/** Static pipeline parameters from the implementation plan. */
const PIPELINE_STATS = [
  { label: 'Daily image volume', value: '50 images/PHC', icon: '📷' },
  { label: 'Upload bandwidth', value: '2 Mbps', icon: '📡' },
  { label: 'AI inference time', value: '~350 ms', icon: '🤖' },
  { label: 'Referral rate', value: '18%', icon: '⚠️' },
  { label: 'Auto-clear rate', value: '82%', icon: '✅' },
  { label: 'Review time / case', value: '<30 s', icon: '👨‍⚕️' },
] as const;

interface PipelineNodeProps {
  icon: string;
  title: string;
  subtitle: string;
  color: string;
  isLast?: boolean;
}

function PipelineNode({ icon, title, subtitle, color, isLast }: PipelineNodeProps) {
  return (
    <div className="flex items-center">
      <div className={`flex flex-col items-center p-3 rounded-xl border-2 ${color} bg-white min-w-[120px]`}>
        <span className="text-2xl">{icon}</span>
        <span className="text-xs font-semibold mt-1 text-center">{title}</span>
        <span className="text-[10px] text-gray-500 text-center">{subtitle}</span>
      </div>
      {!isLast && (
        <div className="mx-1 text-gray-400 text-lg font-bold shrink-0">→</div>
      )}
    </div>
  );
}

export function SimulinkViz() {
  return (
    <div className="space-y-4">
      <Card title="Telemedicine Screening Pipeline">
        <p className="text-sm text-gray-500 mb-4">
          Simulink-modeled workflow for AI-assisted DR screening in rural Primary Health Centres.
        </p>

        {/* Pipeline flow */}
        <div className="overflow-x-auto pb-2">
          <div className="flex items-center min-w-[700px]">
            <PipelineNode
              icon="📷"
              title="PHC Capture"
              subtitle="50 img/day"
              color="border-blue-300"
            />
            <PipelineNode
              icon="📡"
              title="Upload"
              subtitle="2 Mbps link"
              color="border-indigo-300"
            />
            <PipelineNode
              icon="🤖"
              title="AI Server"
              subtitle="~350ms/img"
              color="border-purple-300"
            />
            <PipelineNode
              icon="🔀"
              title="Triage"
              subtitle="Referable?"
              color="border-amber-300"
            />

            {/* Branch */}
            <div className="flex flex-col gap-2 ml-1">
              <div className="flex items-center">
                <span className="text-xs text-gray-400 mr-1 w-14 text-right">No 82%</span>
                <div className="flex flex-col items-center p-2 rounded-xl border-2 border-green-300 bg-green-50 min-w-[100px]">
                  <span className="text-xl">✅</span>
                  <span className="text-xs font-semibold">Auto-clear</span>
                </div>
              </div>
              <div className="flex items-center">
                <span className="text-xs text-gray-400 mr-1 w-14 text-right">Yes 18%</span>
                <div className="flex items-center">
                  <div className="flex flex-col items-center p-2 rounded-xl border-2 border-orange-300 bg-orange-50 min-w-[100px]">
                    <span className="text-xl">👨‍⚕️</span>
                    <span className="text-xs font-semibold">Review</span>
                    <span className="text-[10px] text-gray-500">&lt;30s/case</span>
                  </div>
                  <span className="mx-1 text-gray-400 text-lg font-bold">→</span>
                  <div className="flex flex-col items-center p-2 rounded-xl border-2 border-teal-300 bg-teal-50 min-w-[100px]">
                    <span className="text-xl">📋</span>
                    <span className="text-xs font-semibold">Report</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </Card>

      {/* Key metrics */}
      <Card title="Pipeline Parameters">
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
          {PIPELINE_STATS.map((stat) => (
            <div key={stat.label} className="bg-gray-50 rounded-lg p-3 text-center">
              <div className="text-xl">{stat.icon}</div>
              <div className="text-sm font-bold text-gray-900 mt-1">{stat.value}</div>
              <div className="text-xs text-gray-500">{stat.label}</div>
            </div>
          ))}
        </div>
      </Card>

      {/* Throughput analysis */}
      <Card title="Throughput Analysis">
        <div className="space-y-3 text-sm text-gray-700">
          <div className="bg-blue-50 rounded-lg p-3">
            <p className="font-semibold text-blue-900">Daily Capacity</p>
            <p className="text-blue-700 mt-1">
              With ~350ms per image inference time, a single CPU server can process
              <strong> ~10,000 images/day</strong> — covering <strong>200 PHCs</strong> at 50 images each.
            </p>
          </div>
          <div className="bg-green-50 rounded-lg p-3">
            <p className="font-semibold text-green-900">Specialist Workload Reduction</p>
            <p className="text-green-700 mt-1">
              With 82% auto-cleared, an ophthalmologist reviewing 18% of 50 scans =
              <strong> 9 cases/day × 30s = ~4.5 minutes</strong> per PHC.
            </p>
          </div>
          <div className="bg-amber-50 rounded-lg p-3">
            <p className="font-semibold text-amber-900">Bandwidth Requirement</p>
            <p className="text-amber-700 mt-1">
              At ~500 KB/image (compressed JPEG) × 50 images/day =
              <strong> 25 MB/day</strong> — feasible on 2 Mbps rural links in &lt;2 minutes.
            </p>
          </div>
        </div>
      </Card>
    </div>
  );
}
