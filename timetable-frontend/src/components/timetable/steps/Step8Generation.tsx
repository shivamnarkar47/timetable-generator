import { useState } from 'react';
import { TimetableFormData, ResultsData } from '@/types/timetable';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Play, Loader2, AlertTriangle } from 'lucide-react';
import { toast } from '@/hooks/use-toast';

interface Step8GenerationProps {
  data: any;
  formData: TimetableFormData;
  onUpdate: (data: any) => void;
  onResultsUpdate: (results: ResultsData) => void;
  onGenerate: () => void;
}

export const Step8Generation = ({ 
  formData, 
  onResultsUpdate, 
  onGenerate 
}: Step8GenerationProps) => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [logs, setLogs] = useState<string[]>([]);

  // --- TRANSFORM DATA FOR PYTHON BACKEND ---
  const transformDataForBackend = () => {
    const { welcome, timing, curriculum, infrastructure, faculty, workload, constraints } = formData;

    // 1. Group Subjects by Year for 'subjects' dict
    const subjectsMap: Record<string, any[]> = {};
    const activeClasses = welcome.classes.filter(c => c.selected).map(c => c.name);
    
    activeClasses.forEach(year => {
      subjectsMap[year] = [];
      
      // Theory & Electives
      curriculum.theorySubjects.filter(s => s.year === year).forEach(s => {
        subjectsMap[year].push({
          name: s.name,
          code: s.code,
          type: s.type, // "Theory" or "Elective"
          weekly_load: s.weeklyLoad,
          duration: 1 // Default duration for theory is 1 slot
        });
      });

      // Labs & Tutorials
      curriculum.labSubjects.filter(s => s.year === year).forEach(s => {
        subjectsMap[year].push({
          name: s.name,
          code: s.code,
          type: s.type, // "Lab" or "Tutorial"
          // Calculate total slots needed per week: sessions * duration per session
          weekly_load: s.sessionsPerWeek * s.durationPerSession,
          duration: s.durationPerSession, // e.g. 2 for Labs, 1 for Tuts
          batch_count: s.batchCount,
          is_special: s.isSpecial
        });
      });
    });

    // 2. Flatten Allocations
    const flatAllocations: any[] = [];
    workload.allocations.forEach(alloc => {
      Object.entries(alloc.divisions).forEach(([div, teacherId]) => {
        if (teacherId) {
          flatAllocations.push({
            teacher_id: teacherId,
            subject_name: alloc.subjectName, // This matches 'name' in subjectsMap
            division: div
          });
        }
      });
    });

    // 3. Divisions Mapping (e.g. SE -> [SE-A, SE-B])
    const divisionsMap: Record<string, string[]> = {};
    welcome.classes.filter(c => c.selected).forEach(c => {
      divisionsMap[c.name] = Array.from({length: c.divisions}, (_, i) => `${c.name}-${String.fromCharCode(65+i)}`);
    });

    // PAYLOAD CONSTRUCTION
    return {
      config: {
        slots_per_day: timing.totalSlots,
        recess_index: timing.recessAfterSlot - 1, // Convert 1-based UI to 0-based Backend index
        days: timing.workingDays
      },
      resources: {
        lab_rooms: infrastructure.labRooms,
        theory_rooms: infrastructure.theoryRooms
      },
      subjects: subjectsMap,
      lab_prefs: constraints.labEquipmentMapping,
      home_rooms: constraints.homeRoomAssignments,
      shift_bias: constraints.shiftBias,
      
      faculty: faculty.faculty.map(f => ({
        id: f.id,
        name: f.name,
        role: f.role,
        experience: f.experience,
        shift: f.shift === '9-5' ? '9-5' : '10-6' 
      })),
      
      allocations: flatAllocations,
      divisions: divisionsMap,
      
      // Pass room details for special assignments
      rooms: [
        ...infrastructure.theoryRooms.map(r => ({ name: r, type: 'Classroom' })),
        ...infrastructure.labRooms.map(r => ({ 
            name: r, 
            type: 'Lab',
            // Check if this room is assigned to any special lab subject
            special_assignment: Object.keys(infrastructure.specialAssignments).find(
                key => infrastructure.specialAssignments[key] === r
            ) 
        }))
      ]
    };
  };

  const handleRunAlgorithm = async () => {
    setLoading(true);
    setError(null);
    setLogs(["Preparing data structure...", "Connecting to Solver Engine..."]);

    try {
      const payload = transformDataForBackend();
      console.log("Sending Payload:", JSON.stringify(payload, null, 2)); // Debugging log

      const response = await fetch(`${import.meta.env.VITE_API_URL}/generate-timetable`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!response.ok) {
        const errText = await response.text();
        throw new Error(`Server Error: ${errText}`);
      }

      setLogs(prev => [...prev, "Algorithm Running... (Genetic Evolution in progress)", "Optimizing fitness score..."]);
      
      const result = await response.json();
      
      setLogs(prev => [...prev, "Solution Found!", "Processing Results..."]);

      // Update Result State
      onResultsUpdate({
        totalGaps: 0, 
        unplacedLectures: 0, 
        fitnessScore: 9500, // Dummy score or return from backend
        timetable: result
      });

      toast({
        title: "Timetable Generated Successfully",
        description: "Moving to results view...",
      });

      setTimeout(() => {
        onGenerate(); // Move to Step 9
      }, 1000);

    } catch (err: any) {
      console.error(err);
      setError(err.message || "Failed to generate timetable");
      setLogs(prev => [...prev, `ERROR: ${err.message}`]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="form-section animate-slide-up text-center py-10">
      <div className="mb-8">
        <div className="w-20 h-20 rounded-full gradient-navy flex items-center justify-center mx-auto mb-6 shadow-xl animate-pulse-slow">
          {loading ? (
            <Loader2 className="w-10 h-10 text-gold animate-spin" />
          ) : (
            <Play className="w-10 h-10 text-gold ml-1" />
          )}
        </div>
        <h2 className="text-3xl font-display font-bold mb-2">Ready to Generate</h2>
        <p className="text-muted-foreground max-w-md mx-auto">
          We have collected all configuration, curriculum, and constraints. 
          The Genetic Algorithm will now attempt to find the optimal schedule.
        </p>
      </div>

      {!loading && !error && (
        <Button size="lg" onClick={handleRunAlgorithm} className="px-8 py-6 text-lg shadow-lg hover:shadow-xl hover:scale-105 transition-all">
          Start Generation Process
        </Button>
      )}

      {(loading || logs.length > 0) && (
        <Card className="max-w-xl mx-auto mt-8 p-4 text-left bg-black/5 border-black/10">
          <div className="space-y-2 font-mono text-sm">
            {logs.map((log, i) => (
              <div key={i} className="flex items-center gap-2">
                <span className="text-green-600">➜</span>
                <span>{log}</span>
              </div>
            ))}
          </div>
        </Card>
      )}

      {error && (
        <div className="max-w-xl mx-auto mt-8 p-4 rounded-lg bg-red-50 text-red-600 border border-red-200 flex items-center gap-3">
          <AlertTriangle className="w-5 h-5" />
          <p className="flex-1 text-sm">{error}</p>
          <Button variant="outline" size="sm" onClick={() => setError(null)} className="ml-auto border-red-200 hover:bg-red-100">
            Retry
          </Button>
        </div>
      )}
    </div>
  );
};