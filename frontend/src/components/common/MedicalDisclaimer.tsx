export function MedicalDisclaimer() {
  return (
    <div className="bg-amber-50 border border-amber-200 rounded-lg px-4 py-3 text-xs text-amber-800">
      <p className="font-semibold">⚠️ Research Prototype — Not a Medical Device</p>
      <p className="mt-1">
        This is an investigational screening tool for demonstration purposes. It is NOT
        FDA/CE cleared and must NOT be used for clinical diagnostic decisions. All results
        require confirmation by a qualified ophthalmologist.
      </p>
    </div>
  );
}
