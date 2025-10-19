import { useCallback, useEffect, useRef } from 'react';
import jsMind from 'jsmind';
import { DEFAULT_TOPIC } from '../utils/type';
import { useOpenAITaxonomy } from '../hooks/useOpenAI';
import { useStore } from '../store/useStore';
import { FaStar } from "react-icons/fa6";

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
    const apiKey = import.meta.env.VITE_OPENAI_API_KEY as string;
    const result = await callOpenAI(apiKey);
    if (result) {
      console.log('OpenAI Analysis:', result);
    }
  };

  return (
    <section id="mindmap-panel" className={`tab-panel ${active ? 'active' : ''}`} role="tabpanel" aria-labelledby="mindmap-tab" hidden={!active}>
      <div style={{ flexDirection: 'row' }}>
        <div className="project-context-banner" aria-live="polite">
          <div className="project-context-heading" style={{ flexDirection: 'row'}}>
            <span className="project-context-pill">Currently viewing</span>
            {contextText ? (
              <h5 className="project-context-title">{contextText}</h5>
            ) : null}
          </div>
          <div style={{ flexDirection: 'row'}}>
            {contextDescription ? (
              <p className="project-context-description">{contextDescription}</p>
            ) : null}
            {contextText && (
            <button
              type="button"
              className="mindmap-explore-button"
              onClick={handleAnalyzeTaxonomy}
            >
              <FaStar />
              Explore
            </button>
            )}
          </div>
        </div>
      </div>
      <div id="jsmind_container" ref={containerRef} />
      <span className="status" id="status">{statusText}</span>
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