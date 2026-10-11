import React, { useState, useEffect, useRef } from "react";
import TopHeader from "./components/TopHeader";
import WorkflowStepper from "./components/WorkflowStepper";
import InputSection from "./components/InputSection";
import AnalyzeSection from "./components/AnalyzeSection";
import ResultsSection from "./components/ResultsSection";
import ExplanationSection from "./components/ExplanationSection";
import {
  fetchAttackOptions,
  fetchSampleData,
  fetchSampleCameraFile,
  fetchSampleRadarFile,
  runEvaluation,
} from "./lib/api";

export default function App() {
  // Input State
  const [instruction, setInstruction] = useState(
    "Stop behind the white delivery van up ahead."
  );
  const [cameraImage, setCameraImage] = useState(null);
  const [cameraPreview, setCameraPreview] = useState(null);
  const [radarFile, setRadarFile] = useState(null);
  const [radarFileName, setRadarFileName] = useState("");

  // Attack State
  const [promptAttack, setPromptAttack] = useState("none");
  const [cameraAttack, setCameraAttack] = useState("none");
  const [radarAttack, setRadarAttack] = useState("none");

  // Metadata State
  const [attackOptions, setAttackOptions] = useState({
    prompt_attacks: [],
    camera_attacks: [],
    radar_attacks: [],
  });
  const [sampleData, setSampleData] = useState(null);

  // Execution State
  const [isEvaluating, setIsEvaluating] = useState(false);
  const [evaluationResult, setEvaluationResult] = useState(null);
  const [currentStep, setCurrentStep] = useState("input");
  const [error, setError] = useState("");

  const resultsRef = useRef(null);
  const inputRef = useRef(null);

  // 1. Initial Load: Fetch attack options & sample metadata
  useEffect(() => {
    async function init() {
      try {
        const [attacks, samples] = await Promise.allSettled([
          fetchAttackOptions(),
          fetchSampleData(),
        ]);
        if (attacks.status === "fulfilled") {
          setAttackOptions(attacks.value);
        }
        if (samples.status === "fulfilled") {
          setSampleData(samples.value);
          // Set sample camera preview if available
          if (samples.value?.camera_image_base64) {
            setCameraPreview(
              `data:image/jpeg;base64,${samples.value.camera_image_base64}`
            );
          }
          if (samples.value?.radar_filename) {
            setRadarFileName(samples.value.radar_filename);
          }
        }
      } catch (err) {
        console.error("Failed to load initial metadata:", err);
      }
    }
    init();
  }, []);

  // 2. Quick Action: Load Sample Scenario
  const handleLoadSample = async () => {
    setError("");
    try {
      setInstruction("Stop behind the white delivery van up ahead.");
      setPromptAttack("none");
      setCameraAttack("none");
      setRadarAttack("none");

      if (sampleData?.camera_image_base64) {
        setCameraPreview(
          `data:image/jpeg;base64,${sampleData.camera_image_base64}`
        );
      } else {
        const camFile = await fetchSampleCameraFile();
        setCameraImage(camFile);
        setCameraPreview(URL.createObjectURL(camFile));
      }

      if (sampleData?.radar_filename) {
        setRadarFileName(sampleData.radar_filename);
      } else {
        const radFile = await fetchSampleRadarFile();
        setRadarFile(radFile);
        setRadarFileName(radFile.name);
      }
    } catch (err) {
      console.warn("Could not load sample files:", err);
    }
  };

  // 3. Quick Action: Use Sample Camera
  const handleUseSampleCamera = async () => {
    setError("");
    try {
      if (sampleData?.camera_image_base64) {
        setCameraPreview(
          `data:image/jpeg;base64,${sampleData.camera_image_base64}`
        );
        setCameraImage(null); // Backend falls back to server-side sample image
      } else {
        const camFile = await fetchSampleCameraFile();
        setCameraImage(camFile);
        setCameraPreview(URL.createObjectURL(camFile));
      }
    } catch (err) {
      setError("Failed to load sample camera image: " + err.message);
    }
  };

  // 4. Quick Action: Use Sample Radar
  const handleUseSampleRadar = async () => {
    setError("");
    try {
      if (sampleData?.radar_filename) {
        setRadarFileName(sampleData.radar_filename);
        setRadarFile(null); // Backend falls back to server-side sample radar
      } else {
        const radFile = await fetchSampleRadarFile();
        setRadarFile(radFile);
        setRadarFileName(radFile.name);
      }
    } catch (err) {
      setError("Failed to load sample radar file: " + err.message);
    }
  };

  // 5. Select Preset Instruction
  const handleSelectPreset = (preset) => {
    setInstruction(preset.instruction);
    setError("");
  };

  // 6. Execute Evaluation
  const handleSubmit = async () => {
    if (!instruction.trim()) {
      setError("Please provide a passenger navigation instruction.");
      return;
    }

    setError("");
    setIsEvaluating(true);
    setCurrentStep("analyze");

    try {
      const result = await runEvaluation({
        instruction,
        cameraImage,
        radarFile,
        promptAttack,
        cameraAttack,
        radarAttack,
      });

      setEvaluationResult(result);
      setCurrentStep("results");

      // Smooth scroll to results
      setTimeout(() => {
        resultsRef.current?.scrollIntoView({ behavior: "smooth" });
      }, 100);
    } catch (err) {
      setError(err.message || "Failed to evaluate scenario.");
      setCurrentStep("input");
    } finally {
      setIsEvaluating(false);
    }
  };

  // 7. Reset Form
  const handleReset = () => {
    setEvaluationResult(null);
    setCurrentStep("input");
    setError("");
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  // 8. Stepper Click
  const handleStepClick = (stepId) => {
    setCurrentStep(stepId);
    if (stepId === "input") {
      inputRef.current?.scrollIntoView({ behavior: "smooth" });
    } else if (stepId === "results" || stepId === "explanation") {
      resultsRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  };

  return (
    <div className="min-h-screen bg-canvas text-slate-100 flex flex-col font-sans selection:bg-brand-500 selection:text-white">
      {/* Top Header */}
      <TopHeader
        onLoadSample={handleLoadSample}
        onReset={handleReset}
        isEvaluating={isEvaluating}
        hasResults={Boolean(evaluationResult)}
      />

      {/* Main Container */}
      <main className="flex-1 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 w-full">
        {/* Workflow Stepper: Input -> Analyze -> Results -> Explanation */}
        <WorkflowStepper
          currentStep={currentStep}
          hasResults={Boolean(evaluationResult)}
          onStepClick={handleStepClick}
        />

        {/* STEP 1: INPUT SECTION */}
        <div ref={inputRef}>
          <InputSection
            instruction={instruction}
            setInstruction={setInstruction}
            cameraImage={cameraImage}
            setCameraImage={setCameraImage}
            cameraPreview={cameraPreview}
            setCameraPreview={setCameraPreview}
            radarFile={radarFile}
            setRadarFile={setRadarFile}
            radarFileName={radarFileName}
            setRadarFileName={setRadarFileName}
            promptAttack={promptAttack}
            setPromptAttack={setPromptAttack}
            cameraAttack={cameraAttack}
            setCameraAttack={setCameraAttack}
            radarAttack={radarAttack}
            setRadarAttack={setRadarAttack}
            attackOptions={attackOptions}
            sampleData={sampleData}
            onUseSampleCamera={handleUseSampleCamera}
            onUseSampleRadar={handleUseSampleRadar}
            onSelectPreset={handleSelectPreset}
            onSubmit={handleSubmit}
            isEvaluating={isEvaluating}
            error={error}
          />
        </div>

        {/* STEP 2: ANALYZE SECTION (Telemetric Progress) */}
        {isEvaluating && <AnalyzeSection />}

        {/* STEP 3 & 4: RESULTS & EXPLANATION SECTIONS */}
        {evaluationResult && !isEvaluating && (
          <div ref={resultsRef} className="mt-12 space-y-12">
            <div className="border-t border-surface-border pt-8">
              <ResultsSection result={evaluationResult} />
            </div>

            <div className="border-t border-surface-border pt-8">
              <ExplanationSection
                result={evaluationResult}
                onReset={handleReset}
              />
            </div>
          </div>
        )}
      </main>

      {/* Modern Research Footer */}
      <footer className="border-t border-surface-border bg-canvas/80 py-6 mt-auto">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-3 text-xs text-slate-500 font-sans">
          <div>
            AegisDrive • Adversarial Robustness Verification Framework for Autonomous Vehicles
          </div>
          <div className="font-mono text-[11px] text-slate-400">
            nuScenes-mini & Talk2Car Multimodal Foundation
          </div>
        </div>
      </footer>
    </div>
  );
}
