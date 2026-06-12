'use client';

import { ChevronLeft, Loader2, Sparkles } from 'lucide-react';
import { useState } from 'react';
import { toast } from 'sonner';
import { Button } from '@/src/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/src/components/ui/dialog';
import { cn } from '@/src/lib/utils';
import {
  type BackendMode,
  type GenerateTaxonomyResponse,
  type ReasoningEffort,
  useGenerateTaxonomyMutation,
} from '../hooks/use-generate-taxonomy-mutation';

type Step = 'form' | 'confirm';

interface GenerateTaxonomyDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  /** Receives the generated taxonomy AND the overview it came from, so the
   * caller can persist the project brief (Part 13 L-B: editable later via
   * "Edit Brief & Taxonomy"). */
  onSuccess?: (result: GenerateTaxonomyResponse, overview: string) => void;
  /** Prefill for the overview field — the persisted project brief. */
  initialOverview?: string;
}

const fieldClass =
  'w-full rounded-md border border-input bg-transparent px-3 py-2 text-sm shadow-xs outline-none transition-[color,box-shadow] placeholder:text-muted-foreground focus-visible:border-ring focus-visible:ring-[3px] focus-visible:ring-ring/50 disabled:opacity-50';

export function GenerateTaxonomyDialog({
  open,
  onOpenChange,
  onSuccess,
  initialOverview,
}: GenerateTaxonomyDialogProps) {
  const [step, setStep] = useState<Step>('form');
  // Derived-value pattern (no set-state-in-effect): the field shows the
  // persisted brief until the designer types; typing takes over, and closing
  // resets so the next open prefills fresh.
  const [typedOverview, setTypedOverview] = useState<string | null>(null);
  const projectOverview = typedOverview ?? initialOverview ?? '';
  const [reasoningEffort, setReasoningEffort] = useState<ReasoningEffort>('medium');
  const [mode, setMode] = useState<BackendMode>('vllm');

  const { mutate, isPending, error, reset } = useGenerateTaxonomyMutation();

  const handleFormSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setStep('confirm');
  };

  const handleConfirm = () => {
    mutate(
      { project_overview: projectOverview, reasoning_effort: reasoningEffort, mode },
      {
        onSuccess: (result) => {
          onSuccess?.(result, projectOverview);
          toast.success(`Taxonomy generated — ${result.aspects.length} aspect${result.aspects.length !== 1 ? 's' : ''} created.`);
          onOpenChange(false);
        },
      }
    );
  };

  const handleOpenChange = (next: boolean) => {
    if (!isPending) {
      onOpenChange(next);
      if (!next) {
        reset();
        setStep('form');
        setTypedOverview(null);
      }
    }
  };

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogContent className="sm:max-w-lg">
        {step === 'form' ? (
          <>
            <DialogHeader>
              <DialogTitle>Generate Taxonomy</DialogTitle>
              <DialogDescription>
                Describe your project to generate a structured design taxonomy.
              </DialogDescription>
            </DialogHeader>

            <form onSubmit={handleFormSubmit} className="space-y-4">
              <div className="space-y-1.5">
                <label className="text-sm font-medium" htmlFor="project-overview">
                  Project Overview
                </label>
                <textarea
                  id="project-overview"
                  className={cn(fieldClass, 'min-h-[120px] resize-y')}
                  placeholder="Describe your project, goals, and context..."
                  value={projectOverview}
                  onChange={(e: React.ChangeEvent<HTMLTextAreaElement>) =>
                    setTypedOverview(e.target.value)
                  }
                  required
                  disabled={isPending}
                />
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div className="space-y-1.5">
                  <label className="text-sm font-medium" htmlFor="reasoning-effort">
                    Reasoning Effort
                  </label>
                  <select
                    id="reasoning-effort"
                    className={cn(fieldClass, 'h-9 cursor-pointer')}
                    value={reasoningEffort}
                    onChange={(e) => setReasoningEffort(e.target.value as ReasoningEffort)}
                    disabled={isPending}
                  >
                    <option value="low">Low</option>
                    <option value="medium">Medium</option>
                    <option value="high">High</option>
                  </select>
                </div>

                <div className="space-y-1.5">
                  <label className="text-sm font-medium" htmlFor="mode">
                    Backend
                  </label>
                  <select
                    id="mode"
                    className={cn(fieldClass, 'h-9 cursor-pointer')}
                    value={mode}
                    onChange={(e) => setMode(e.target.value as BackendMode)}
                    disabled={isPending}
                  >
                    <option value="openai">OpenAI</option>
                    <option value="vllm">vLLM</option>
                  </select>
                </div>
              </div>

              {error ? (
                <p className="rounded-lg bg-destructive/10 px-3 py-2 text-xs font-medium text-destructive">
                  {error.message}
                </p>
              ) : null}

              <DialogFooter>
                <Button
                  type="button"
                  variant="outline"
                  onClick={() => handleOpenChange(false)}
                  disabled={isPending}
                >
                  Cancel
                </Button>
                <Button type="submit" disabled={isPending || !projectOverview.trim()}>
                  <Sparkles className="mr-1.5 h-3.5 w-3.5" />
                  Next
                </Button>
              </DialogFooter>
            </form>
          </>
        ) : (
          <>
            <DialogHeader>
              <DialogTitle>Confirm Generation</DialogTitle>
              <DialogDescription>
                Review your settings before generating the taxonomy.
              </DialogDescription>
            </DialogHeader>

            <div className="space-y-3 rounded-lg border bg-muted/40 p-4 text-sm">
              <div className="flex justify-between gap-4">
                <span className="text-muted-foreground">Reasoning</span>
                <span className="font-medium capitalize">{reasoningEffort}</span>
              </div>
              <div className="flex justify-between gap-4">
                <span className="text-muted-foreground">Backend</span>
                <span className="font-medium">{mode === 'openai' ? 'OpenAI' : 'vLLM'}</span>
              </div>
              <div className="space-y-1">
                <span className="text-muted-foreground">Project overview</span>
                <p className="text-foreground leading-relaxed line-clamp-4">{projectOverview.trim()}</p>
              </div>
            </div>

            {error ? (
              <p className="rounded-lg bg-destructive/10 px-3 py-2 text-xs font-medium text-destructive">
                {error.message}
              </p>
            ) : null}

            <DialogFooter>
              <Button
                type="button"
                variant="outline"
                onClick={() => setStep('form')}
                disabled={isPending}
              >
                <ChevronLeft className="mr-1.5 h-3.5 w-3.5" />
                Back
              </Button>
              <Button type="button" onClick={handleConfirm} disabled={isPending}>
                {isPending ? (
                  <Loader2 className="mr-1.5 h-3.5 w-3.5 animate-spin" />
                ) : (
                  <Sparkles className="mr-1.5 h-3.5 w-3.5" />
                )}
                {isPending ? 'Generating…' : 'Confirm & Generate'}
              </Button>
            </DialogFooter>
          </>
        )}
      </DialogContent>
    </Dialog>
  );
}
