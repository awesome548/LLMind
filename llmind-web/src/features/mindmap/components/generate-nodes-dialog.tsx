'use client';

import { ChevronLeft, Loader2, Zap } from 'lucide-react';
import { useState } from 'react';
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
import type { GenerateNodesParams } from '../hooks/use-generate-nodes-mutation';

type BackendMode = 'openai' | 'vllm';
type ReasoningEffort = 'low' | 'medium' | 'high';

export interface GenerateNodesDialogParams {
  additionalContext: string;
  reasoningEffort: ReasoningEffort;
  mode: BackendMode;
}

interface GenerateNodesDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onConfirm: (params: Pick<GenerateNodesParams, 'description' | 'mode' | 'reasoningEffort'>) => void;
  isPending: boolean;
  selectedTopic: string;
}

const fieldClass =
  'w-full rounded-md border border-input bg-transparent px-3 py-2 text-sm shadow-xs outline-none transition-[color,box-shadow] placeholder:text-muted-foreground focus-visible:border-ring focus-visible:ring-[3px] focus-visible:ring-ring/50 disabled:opacity-50';

type Step = 'form' | 'confirm';

export function GenerateNodesDialog({
  open,
  onOpenChange,
  onConfirm,
  isPending,
  selectedTopic,
}: GenerateNodesDialogProps) {
  const [step, setStep] = useState<Step>('form');
  const [additionalContext, setAdditionalContext] = useState('');
  const [reasoningEffort, setReasoningEffort] = useState<ReasoningEffort>('medium');
  const [mode, setMode] = useState<BackendMode>('openai');

  const handleFormSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setStep('confirm');
  };

  const handleConfirm = () => {
    onConfirm({
      description: additionalContext.trim() || undefined,
      mode,
      reasoningEffort,
    });
  };

  const handleOpenChange = (next: boolean) => {
    if (!isPending) {
      onOpenChange(next);
      if (!next) {
        setStep('form');
      }
    }
  };

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogContent className="sm:max-w-lg">
        {step === 'form' ? (
          <>
            <DialogHeader>
              <DialogTitle>Generate Nodes</DialogTitle>
              <DialogDescription>
                Expand <span className="font-medium text-foreground">"{selectedTopic}"</span> with
                AI-generated child nodes.
              </DialogDescription>
            </DialogHeader>

            <form onSubmit={handleFormSubmit} className="space-y-4">
              <div className="space-y-1.5">
                <label className="text-sm font-medium" htmlFor="additional-context">
                  Additional Context{' '}
                  <span className="text-muted-foreground font-normal">(optional)</span>
                </label>
                <textarea
                  id="additional-context"
                  className={cn(fieldClass, 'min-h-[100px] resize-y')}
                  placeholder="Add any extra context or constraints for node generation..."
                  value={additionalContext}
                  onChange={(e) => setAdditionalContext(e.target.value)}
                  disabled={isPending}
                />
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div className="space-y-1.5">
                  <label className="text-sm font-medium" htmlFor="nodes-reasoning-effort">
                    Reasoning Effort
                  </label>
                  <select
                    id="nodes-reasoning-effort"
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
                  <label className="text-sm font-medium" htmlFor="nodes-mode">
                    Backend
                  </label>
                  <select
                    id="nodes-mode"
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

              <DialogFooter>
                <Button
                  type="button"
                  variant="outline"
                  onClick={() => handleOpenChange(false)}
                  disabled={isPending}
                >
                  Cancel
                </Button>
                <Button type="submit">
                  <Zap className="mr-1.5 h-3.5 w-3.5 fill-current" />
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
                Review your settings before generating nodes.
              </DialogDescription>
            </DialogHeader>

            <div className="space-y-3 rounded-lg border bg-muted/40 p-4 text-sm">
              <div className="flex justify-between gap-4">
                <span className="text-muted-foreground">Target node</span>
                <span className="font-medium text-right">{selectedTopic}</span>
              </div>
              <div className="flex justify-between gap-4">
                <span className="text-muted-foreground">Reasoning</span>
                <span className="font-medium capitalize">{reasoningEffort}</span>
              </div>
              <div className="flex justify-between gap-4">
                <span className="text-muted-foreground">Backend</span>
                <span className="font-medium">{mode === 'openai' ? 'OpenAI' : 'vLLM'}</span>
              </div>
              {additionalContext.trim() ? (
                <div className="space-y-1">
                  <span className="text-muted-foreground">Additional context</span>
                  <p className="text-foreground leading-relaxed line-clamp-4">{additionalContext.trim()}</p>
                </div>
              ) : null}
            </div>

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
                  <Zap className="mr-1.5 h-3.5 w-3.5 fill-current" />
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
