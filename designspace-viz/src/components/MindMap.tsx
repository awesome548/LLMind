import { useCallback, useEffect, useRef, useState } from 'react';
import jsMind from 'jsmind';
import { DEFAULT_TOPIC } from '../types/taxonomy';
import { useOpenAITaxonomy } from '../hooks/useOpenAI';
import { useStore } from '../store/useStore';
import { FaStar, FaRegCircleXmark, FaPlus, FaCheck } from "react-icons/fa6";
import type { AddNodeResponse } from '../types/chatCompletion';

const palette = [
  { background: '#4a63ff', color: '#ffffff', size: 20, weight: '600', style: 'normal' },
  { background: '#eef2ff', color: '#1f2937', size: 18, weight: '600', style: 'normal' },
  { background: '#f5f7ff', color: '#1f2937', size: 17, weight: '500', style: 'normal' },
  { background: '#ffffff', color: '#1f2937', size: 16, weight: '500', style: 'normal' },
  { background: '#f9fafc', color: '#1f2937', size: 15, weight: '500', style: 'normal' },
  { background: '#f3f4f6', color: '#1f2937', size: 15, weight: '500', style: 'normal' }
];

export function MindMap({
  active,
  mind,
  onSelect,
  statusText,
}: {
  active: boolean;
  mind: any;
  onSelect: (topic: string, lineage: string[], userInitiated: boolean) => void;
  statusText: string;
}) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const jmRef = useRef<any>(null);
  const stylingScheduled = useRef(false);
  const selectedNodeIdRef = useRef<string | null>(null);
  
  const [isLoading, setIsLoading] = useState(false);
  const [showModal, setShowModal] = useState(false);
  const [suggestedNodes, setSuggestedNodes] = useState<AddNodeResponse[]>([]);
  const [selectedSuggestions, setSelectedSuggestions] = useState<Set<string>>(new Set());
  const modalTitleId = 'mindmap-modal-title';
  const modalDescriptionId = 'mindmap-modal-description';
  
  // Get setJmRef from store to sync
  const setJmRef = useStore(state => state.setJmRef);
  const contextText = useStore(state => state.contextText);
  const contextDescription = useStore(state => state.contextDescription);

  const scheduleStyleRefresh = useCallback(() => {
    if (stylingScheduled.current || !jmRef.current) return;
    stylingScheduled.current = true;
    requestAnimationFrame(() => {
      applyLevelStyling(jmRef.current);
      stylingScheduled.current = false;
    });
  }, []);

  const handleNodeSelection = useCallback((nodeRef: any, userInitiated: boolean) => {
    const jm = jmRef.current;
    if (!jm) return;

    const resolveNode = (candidate: any) => {
      if (!candidate) return null;
      if (typeof candidate === 'string' && candidate.trim()) {
        return jm.get_node(candidate.trim()) ?? findNodeByTopic(jm, candidate);
      }
      if (candidate && typeof candidate === 'object') {
        if (candidate.id) {
          return jm.get_node(candidate.id) || candidate;
        }
        if (candidate.topic) {
          return findNodeByTopic(jm, candidate.topic);
        }
      }
      return null;
    };

    const resolvedNode = resolveNode(nodeRef);

    if (resolvedNode) {
      const resolvedId = String(resolvedNode.id);
      if (selectedNodeIdRef.current !== resolvedId) {
        const currentlySelected =
          typeof jm.get_selected_node === 'function' ? jm.get_selected_node() : null;
        if (!currentlySelected || currentlySelected.id !== resolvedId) {
          jm.select_node(resolvedId);
        }
      }
      selectedNodeIdRef.current = resolvedId;
      const topic =
        normalizeTopic(resolvedNode.topic) ||
        normalizeTopic(resolvedNode?.data?.topic) ||
        DEFAULT_TOPIC;
      const lineage = collectLineage(resolvedNode);
      onSelect(topic, lineage.length ? lineage : [topic], userInitiated);
      return;
    }

    const fallbackTopic =
      typeof nodeRef === 'string' && nodeRef.trim() ? nodeRef.trim() : DEFAULT_TOPIC;
    onSelect(fallbackTopic, [fallbackTopic], userInitiated);
  }, [onSelect]);

  // init jsMind once
  useEffect(() => {
    if (!active || !containerRef.current || jmRef.current) return;
    jmRef.current = new jsMind({
      container: 'jsmind_container',
      theme: 'primary',
      mode: 'full',
      editable: true,
      view:{
        draggable: true,
      },
      shortcut:{
        enable: true,
        mapping: {
          addchild : [45, 1024+13],
        }
      }
    });
    // Store in Zustand
    setJmRef(jmRef.current);

    const mindEvents = (jsMind as any)?.event_type || {};
    jmRef.current.add_event_listener((type: any, data: any) => {
      if (data?.evt === 'select_node' && data?.node) {
        handleNodeSelection(data.node, true);
      }
      if ([mindEvents.show, mindEvents.resize, mindEvents.layout, mindEvents.edit].includes(type)) {
        scheduleStyleRefresh();
      }
    });
  }, [active, jmRef, setJmRef, handleNodeSelection, scheduleStyleRefresh]);

  // show/update mind
  useEffect(() => {
    if (!jmRef.current || !mind) return;
    jmRef.current.show(mind);

    // >>> collapse everything below first-level nodes
    // run on next frame so DOM is ready in all browsers
    requestAnimationFrame(() => collapseBelowFirstLevel(jmRef.current));

    const rootId = mind?.data?.id;
    selectedNodeIdRef.current = null;
    handleNodeSelection(rootId, true);
    scheduleStyleRefresh();
    console.log('Mind map updated', jmRef.current.get_data('node_array'));
  }, [handleNodeSelection, mind, scheduleStyleRefresh]);

  useEffect(() => {
    if (!active || !jmRef.current) return;
    requestAnimationFrame(() => {
      if (selectedNodeIdRef.current) {
        jmRef.current.select_node(selectedNodeIdRef.current);
      }
      jmRef.current.resize?.();
      scheduleStyleRefresh();
    });
  }, [active, scheduleStyleRefresh, jmRef]);

  const { callOpenAI } = useOpenAITaxonomy( jmRef );
  
  const handleAnalyzeTaxonomy = async () => {
    const hasEnv = Boolean(import.meta.env.VITE_OPENAI_API_KEY);
    if (!hasEnv) {
      console.error('OpenAI API key not configured in environment variables.');
      return;
    }
    
    setIsLoading(true);
    const apiKey = import.meta.env.VITE_OPENAI_API_KEY as string;
    
    try {
      const jm = jmRef.current;
      const selected =
        (typeof jm?.get_selected_node === 'function' ? jm.get_selected_node() : null) ||
        (selectedNodeIdRef.current && typeof jm?.get_node === 'function'
          ? jm.get_node(selectedNodeIdRef.current)
          : null) ||
        jm?.mind?.root ||
        null;

      const selectedTopic =
        normalizeTopic(selected?.topic) ||
        DEFAULT_TOPIC;
      const focusNode = selected
        ? {
            id: selected?.id != null ? String(selected.id) : 'root',
            topic: selectedTopic,
          }
        : {
            id: 'root',
            topic: DEFAULT_TOPIC,
          };

      const result = await callOpenAI(apiKey, { focusNode });
      if (result && Array.isArray(result)) {
        console.log('OpenAI Analysis:', result);
        setSuggestedNodes(result);
        setSelectedSuggestions(new Set());
        setShowModal(true);
      }
    } catch (error) {
      console.error('Error analyzing taxonomy:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const toggleSuggestionSelection = (nodeId: string) => {
    setSelectedSuggestions(prev => {
      const newSet = new Set(prev);
      if (newSet.has(nodeId)) {
        newSet.delete(nodeId);
      } else {
        newSet.add(nodeId);
      }
      return newSet;
    });
  };

  const addSelectedNodesToMindMap = () => {
    if (!jmRef.current) return;
    
    suggestedNodes.forEach(node => {
      if (selectedSuggestions.has(node.node_id)) {
        // Check if parent node exists
        const parentNode = jmRef.current.get_node(node.parent_node);
        if (parentNode) {
          // Add the node to the mind map
          jmRef.current.add_node(
            node.parent_node,
            node.node_id,
            node.topic,
            {},
            undefined // direction - let jsMind decide based on layout
          );
        } else {
          console.warn(`Parent node ${node.parent_node} not found for ${node.node_id}`);
        }
      }
    });
    
    // Refresh styling after adding nodes
    scheduleStyleRefresh();
    
    // Close modal and reset selections
    setShowModal(false);
    setSelectedSuggestions(new Set());
    setSuggestedNodes([]);
  };

  useEffect(() => {
    if (!showModal) return;
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        event.preventDefault();
        setShowModal(false);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [showModal]);

  return (
    <section id="mindmap-panel" className={`tab-panel ${active ? 'active' : ''}`} role="tabpanel" aria-labelledby="mindmap-tab" hidden={!active}>
      <div style={{ flexDirection: 'row' }}>
        <div className="project-context-banner" aria-live="polite">
          <div className="project-context-heading" style={{ flexDirection: 'row', alignItems: 'center' }}>
            {contextText ? (
              <>
              <h5 className="project-context-title">{contextText}</h5>
            <span className="project-context-pill">Currently viewing</span>
              </>
            ) : <p className='project-context-description'>No node selected</p>}
          </div>
          <div style={{ flexDirection: 'row'}}>
            {contextDescription ? (
              <p className="project-context-description">{contextDescription}</p>
            ) : null}
            {contextText && (
            <button
              type="button"
              className={`mindmap-explore-button ${isLoading ? 'mindmap-explore-button--loading' : ''}`}
              onClick={handleAnalyzeTaxonomy}
              disabled={isLoading}
            >
              {isLoading ? (
                <>
                  <span className="spinner" aria-hidden="true" />
                  <span className="mindmap-explore-label">Analyzing</span>
                </>
              ) : (
                <>
                  <span className="mindmap-explore-icon" aria-hidden="true">
                    <FaStar />
                  </span>
                  <span className="mindmap-explore-text">
                    Explore with AI
                  </span>
                </>
              )}
            </button>
            )}
          </div>
        </div>
      </div>
      <div id="jsmind_container" ref={containerRef} />
      <span className="status" id="status">{statusText}</span>

      {/* Modal for suggested nodes */}
      {showModal && (
        <div
          className="mindmap-modal-overlay"
          role="dialog"
          aria-modal="true"
          aria-labelledby={modalTitleId}
          aria-describedby={modalDescriptionId}
          onClick={() => setShowModal(false)}
        >
          <div
            className="mindmap-modal"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="mindmap-modal-header">
              <div className="mindmap-modal-heading">
                <span className="mindmap-modal-kicker">AI suggestions</span>
                <h3 className="mindmap-modal-title" id={modalTitleId}>
                  Enrich your mind map
                </h3>
                <p className="mindmap-modal-description" id={modalDescriptionId}>
                  Select the topics you would like to add. We will attach them to their recommended parent nodes.
                </p>
              </div>
              <button
                className="mindmap-modal-close"
                onClick={() => setShowModal(false)}
                aria-label="Close modal"
              >
                <FaRegCircleXmark />
              </button>
            </div>
            
            <div className="mindmap-modal-body">
              <div className="mindmap-suggestions-grid">
                {suggestedNodes.map((node) => (
                  <button
                    key={node.node_id}
                    type="button"
                    className={`mindmap-suggestion-card ${selectedSuggestions.has(node.node_id) ? 'is-selected' : ''}`}
                    onClick={() => toggleSuggestionSelection(node.node_id)}
                    aria-pressed={selectedSuggestions.has(node.node_id)}
                    aria-label={`Toggle ${node.topic} under ${node.parent_node}`}
                  >
                    <div className="mindmap-suggestion-card-content">
                      <div className="mindmap-suggestion-checkbox" aria-hidden="true">
                        {selectedSuggestions.has(node.node_id) ? (
                          <FaCheck className="check-icon" />
                        ) : (
                          <FaPlus className="plus-icon" />
                        )}
                      </div>
                      <div className="mindmap-suggestion-text">
                        <p className="mindmap-suggestion-topic">{node.topic}</p>
                        {/* <span className="mindmap-suggestion-parent">
                          Parent: <strong>{node.parent_node}</strong>
                        </span> */}
                      </div>
                    </div>
                  </button>
                ))}
              </div>
            </div>

            <div className="mindmap-modal-footer">
              <button
                type="button"
                className="mindmap-modal-secondary"
                onClick={() => setShowModal(false)}
              >
                Cancel
              </button>
              <button
                type="button"
                className="mindmap-modal-primary"
                onClick={addSelectedNodesToMindMap}
                disabled={selectedSuggestions.size === 0}
              >
                Add Selected ({selectedSuggestions.size})
              </button>
            </div>
          </div>
        </div>
      )}
    </section>
  );
}

function applyLevelStyling(jm: any) {
  const root = jm?.mind?.root;
  if (!root) return;
  const queue = [{ node: root, level: 0 }];
  while (queue.length) {
    const { node, level } = queue.shift()!;
    const c = palette[Math.min(level, palette.length - 1)];
    const signature = `${c.background}|${c.color}|${c.size}|${c.weight}|${c.style}`;
    if (!node.data) node.data = {};
    if (node.data._styleSignature !== signature) {
      jm.set_node_color(node.id, c.background, c.color);
      jm.set_node_font_style(node.id, c.size, c.weight, c.style);
      node.data._styleSignature = signature;
    }
    (node.children || []).forEach((ch: any) => queue.push({ node: ch, level: level + 1 }));
  }
}

function collectLineage(node: any) {
  const topics: string[] = [];
  let cur: any = node;
  let guard = 0;
  while (cur && guard < 100) {
    const t = (cur.topic || cur?.data?.topic || '').trim();
    if (t) topics.unshift(t);
    const parent = cur.parent || cur.parent_node || null;
    if (!parent || parent === cur) break;
    cur = parent; guard++;
  }
  return topics;
}

function findNodeByTopic(jm: any, topic: string) {
  if (!jm?.mind?.nodes || !topic) return null;
  const search = topic.trim().toLowerCase();
  const nodes = jm.mind.nodes as Record<string, any>;
  for (const key of Object.keys(nodes)) {
    const candidate = nodes[key];
    if (typeof candidate?.topic === 'string' && candidate.topic.trim().toLowerCase() === search) {
      return candidate;
    }
  }
  return null;
}

function normalizeTopic(topic: unknown) {
  return typeof topic === 'string' ? topic.trim() : '';
}

function collapseBelowFirstLevel(jm: any) {
  const root = jm?.mind?.root;
  if (!root) return;
  const level1 = root.children || [];
  level1.forEach((n: any) => {
    // collapse this node's subtree (keeps the node itself visible)
    if (n && n.children && n.children.length) {
      jm.collapse_node(n);
    }
  });
}
