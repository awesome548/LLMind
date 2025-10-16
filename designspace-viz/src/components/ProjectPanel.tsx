import { useCallback, useEffect, useState } from 'react';
import type { KeyboardEvent as ReactKeyboardEvent } from 'react';
import type { ProjectDetails } from '../utils/type';
import { truncateText } from '../utils/text';

export function ProjectPanel({
  projects,
  contextText,
  contextDescription,
  statusText,
  isLoading,
}: {
  projects: ProjectDetails[];
  contextText: string;
  contextDescription: string;
  statusText: string;
  isLoading: boolean;
}) {
  const showProjects = !isLoading && projects.length > 0;
  const [selectedProject, setSelectedProject] = useState<ProjectDetails | null>(null);
  const [isDescriptionExpanded, setIsDescriptionExpanded] = useState(false);
  const selectedProjectId = selectedProject?.id ?? selectedProject?.Id ?? null;

  const openProject = useCallback((project: ProjectDetails) => {
    setSelectedProject(project);
    setIsDescriptionExpanded(false);
  }, []);

  const closeProject = useCallback(() => {
    setSelectedProject(null);
    setIsDescriptionExpanded(false);
  }, []);

  const handleKeyDown = useCallback((event: ReactKeyboardEvent<HTMLElement>, project: ProjectDetails) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      openProject(project);
    }
  }, [openProject]);

  useEffect(() => {
    if (!selectedProject) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        event.preventDefault();
        closeProject();
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => {
      window.removeEventListener('keydown', onKeyDown);
    };
  }, [selectedProject, closeProject]);

  return (
    <aside className="project-panel" aria-busy={isLoading}>
      <div>
        <h2 className="panel-title">Related Projects</h2>
        <p className="project-context">Currently viewing: {contextText}</p>
        <p className="project-context-description">{contextDescription}</p>
      </div>
      <div className="project-list-wrapper" aria-live="polite">
        {statusText ? (
          <p className="project-placeholder" role={isLoading ? 'status' : undefined}>
            {statusText}
          </p>
        ) : null}
        {showProjects ? (
          <div className="project-list">
            {projects.map((p, i) => (
              <article
                className="project-card"
                key={`${p.id || p.Id || i}`}
                tabIndex={0}
                role="button"
                aria-label={`View details for ${p.Name || 'Untitled project'}`}
                aria-haspopup="dialog"
                onClick={() => openProject(p)}
                onKeyDown={event => handleKeyDown(event, p)}
              >
                <h3 className="project-title">{p.Name || 'Untitled Project'}</h3>
                {p.Image ? (
                  <div className="project-image-wrapper">
                    <img
                      className="project-image"
                      loading="lazy"
                      src={p.Image}
                      alt={p.Name ? `Preview of ${p.Name}` : 'Project preview image'}
                      referrerPolicy="no-referrer"
                      onError={e => console.error('Failed', e)}
                    />
                  </div>
                ) : null}
                {p.Descriptions ? (
                  <p className="project-description">{truncateText(p.Descriptions, 200)}</p>
                ) : null}
                {p.Details ? (
                  <p className="project-details">{truncateText(p.Details, 100)}</p>
                ) : null}
              </article>
            ))}
          </div>
        ) : null}
      </div>
      {selectedProject ? (
        <div
          className="project-modal-overlay"
          role="dialog"
          aria-modal="true"
          aria-labelledby="project-modal-title"
          onClick={closeProject}
        >
          <div
            className="project-modal"
            onClick={event => event.stopPropagation()}
          >
            <div className="project-modal-header">
              <h3 className="project-modal-title" id="project-modal-title">{selectedProject.Name || 'Untitled Project'}</h3>
              <button
                type="button"
                className="project-modal-close"
                onClick={closeProject}
                aria-label="Close project details"
              >
                &times;
              </button>
            </div>
            {selectedProject.Image ? (
              <img
                className="project-modal-image"
                src={selectedProject.Image}
                alt={selectedProject.Name ? `Preview of ${selectedProject.Name}` : 'Project preview image'}
                referrerPolicy="no-referrer"
              />
            ) : null}
            <div className="project-modal-body">
              {selectedProject.Descriptions ? (() => {
                const fullDescription = selectedProject.Descriptions;
                const truncatedDescription = truncateText(fullDescription, 260);
                const isTruncated = truncatedDescription !== fullDescription;
                const descriptionToShow = isDescriptionExpanded || !isTruncated ? fullDescription : truncatedDescription;
                return (
                  <div className="project-modal-description-block">
                    <p className="project-modal-description">{descriptionToShow}</p>
                    {!isDescriptionExpanded && isTruncated ? (
                      <button
                        type="button"
                        className="project-modal-readmore"
                        onClick={() => setIsDescriptionExpanded(true)}
                      >
                        Read full description
                      </button>
                    ) : null}
                  </div>
                );
              })() : null}
              {selectedProject.Details ? (
                <p className="project-modal-details">{selectedProject.Details}</p>
              ) : null}
            </div>
            <div className="project-modal-actions">
              {selectedProjectId ? (
                <a
                  className="project-external-link"
                  href={`https://awards.mediaarchitecture.org/mab/project/${selectedProjectId}`}
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  View full project
                </a>
              ) : null}
            </div>
          </div>
        </div>
      ) : null}
    </aside>
  );
}
