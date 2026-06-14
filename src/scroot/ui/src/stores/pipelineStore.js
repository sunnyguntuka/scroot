import { create } from 'zustand';
import { persist } from 'zustand/middleware';

export const usePipelineStore = create(
  persist(
    (set, get) => ({
      run: null,           // PipelineRun | null
      config: {            // last-used PipelineConfig (pre-populated on "Run again")
        mode: 'auto_commit',
        record_filter: {
          include_pending: true,
          include_fail_only: false,
          include_all: false,
          agent_id: null,
        },
        auto_commit_thresholds: {
          min_iqs_delta: 0.15,
          min_iqs_floor: 0.70,
        },
      },

      setRun:    (run)    => set({ run }),
      setConfig: (patch)  => set(s => ({ config: { ...s.config, ...patch } })),
      resetRun:  ()       => set({ run: null }),
    }),
    {
      name: 'scroot-pipeline',
      partialize: (s) => ({ config: s.config, run: s.run }),
    }
  )
);
