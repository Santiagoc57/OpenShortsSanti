import React from 'react';
import { Trash2, Activity, FileVideo, Youtube, Link2, Upload, Pencil, Heart, ChevronRight } from 'lucide-react';
import { getApiUrl, projectSourceBadgeClass, projectStatusLabel } from '../utils';

const ProjectsView = ({
  projects,
  visibleProjects,
  projectFilter,
  setProjectFilter,
  favoriteProjectsCount,
  removeAllProjects,
  openSavedProject,
  jobId,
  projectTitleEditJobId,
  projectTitleDraft,
  setProjectTitleDraft,
  saveProjectTitle,
  cancelEditProjectTitle,
  beginEditProjectTitle,
  toggleProjectFavorite,
  removeProject,
  setActiveTab,
  setProjectsViewMode,
  formatProjectDate,
  outputModeLabel
}) => {
  return (
    <div className="max-w-6xl mx-auto py-6 md:py-12 animate-[fadeIn_0.3s_ease-out]">
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-6">
        <div className="flex items-center gap-3">
          <h2 className="text-3xl font-bold text-slate-900 dark:text-white">Mis proyectos</h2>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <button
            type="button"
            onClick={() => setProjectFilter('all')}
            className={`inline-flex min-w-[112px] items-center justify-center gap-2 px-4 py-1.5 rounded-full text-sm font-semibold transition-colors border ${projectFilter === 'all'
              ? 'bg-primary text-white border-primary shadow-sm'
              : 'bg-white dark:bg-slate-800 text-slate-700 dark:text-slate-300 border-slate-200 dark:border-slate-700 hover:bg-slate-50 dark:hover:bg-slate-700'
              }`}
            title="Mostrar todos los proyectos"
          >
            <span className="leading-none">Todos</span>
            <span className={`text-xs px-1.5 py-0.5 rounded-full leading-none ${projectFilter === 'all' ? 'bg-white text-primary' : 'bg-slate-100 text-slate-700 dark:bg-slate-700 dark:text-slate-200'}`}>
              {projects.length}
            </span>
          </button>
          <button
            type="button"
            onClick={() => setProjectFilter('favorites')}
            className={`inline-flex min-w-[128px] items-center justify-center gap-2 px-4 py-1.5 rounded-full text-sm font-semibold transition-colors border ${projectFilter === 'favorites'
              ? 'bg-primary text-white border-primary shadow-sm'
              : 'bg-white dark:bg-slate-800 text-slate-700 dark:text-slate-300 border-slate-200 dark:border-slate-700 hover:bg-slate-50 dark:hover:bg-slate-700'
              }`}
            title="Mostrar solo favoritos"
          >
            <span className="leading-none">Favoritos</span>
            <span className={`text-xs px-1.5 py-0.5 rounded-full leading-none ${projectFilter === 'favorites' ? 'bg-white text-primary' : 'bg-slate-100 text-slate-700 dark:bg-slate-700 dark:text-slate-200'}`}>
              {favoriteProjectsCount}
            </span>
          </button>
          <button
            type="button"
            onClick={removeAllProjects}
            disabled={projects.length === 0}
            className="inline-flex items-center gap-1.5 px-4 py-1.5 rounded-full text-sm font-semibold transition-colors border border-red-300 dark:border-red-700 text-red-600 dark:text-red-300 hover:bg-red-50 dark:hover:bg-red-900/20 disabled:opacity-50 disabled:cursor-not-allowed"
            title="Eliminar todos los proyectos"
          >
            <Trash2 size={14} />
            Eliminar todos
          </button>
        </div>
      </div>

      {visibleProjects.length > 0 ? (
        <>
          <div className="hidden md:grid grid-cols-12 gap-4 px-4 py-2 text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider mb-2">
            <div className="col-span-5">Descripción</div>
            <div className="col-span-2">Origen</div>
            <div className="col-span-3">Tipo de video</div>
            <div className="col-span-1">Ratio</div>
            <div className="col-span-1" />
          </div>
          <div className="space-y-3">
            {visibleProjects.map((project) => (
              <div
                key={project.job_id}
                role="button"
                tabIndex={0}
                onClick={() => {
                  openSavedProject(project);
                  setProjectsViewMode('detail');
                }}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    openSavedProject(project);
                    setProjectsViewMode('detail');
                  }
                }}
                className={`relative cursor-pointer overflow-hidden rounded-2xl border bg-white dark:bg-slate-900/60 p-4 shadow-sm transition-all hover:-translate-y-0.5 hover:shadow-md focus:outline-none focus:ring-2 focus:ring-primary/40 ${project.job_id === jobId
                  ? 'border-primary/60 ring-2 ring-primary/20'
                  : 'border-slate-200 dark:border-slate-700'
                  }`}
              >
                <div className="grid grid-cols-1 md:grid-cols-12 gap-4 items-center">
                  <div className="col-span-1 md:col-span-5 flex items-start gap-4">
                    <div className="relative w-24 h-24 md:w-20 md:h-20 rounded-lg overflow-hidden border border-slate-200 dark:border-slate-700 bg-slate-100 dark:bg-slate-800 flex-shrink-0">
                      {project.thumbnail_url ? (
                        <img src={getApiUrl(project.thumbnail_url)} alt={project.title || 'Proyecto'} className="w-full h-full object-cover" />
                      ) : project.preview_video_url ? (
                        <video src={getApiUrl(project.preview_video_url)} className="w-full h-full object-cover" muted playsInline preload="metadata" />
                      ) : (
                        <div className="w-full h-full flex items-center justify-center">
                          {project.status === 'processing' ? (
                            <Activity size={26} className="text-slate-400 animate-spin" />
                          ) : (
                            <FileVideo size={26} className="text-slate-400" />
                          )}
                        </div>
                      )}
                      <div className="absolute bottom-1 right-1 text-[10px] px-1.5 py-0.5 rounded bg-black/70 text-white backdrop-blur-sm">
                        {outputModeLabel(project.ratio)}
                      </div>
                    </div>
                    <div className="py-1 min-w-0">
                      {projectTitleEditJobId === project.job_id ? (
                        <div className="flex items-center gap-2">
                          <input
                            type="text"
                            value={projectTitleDraft}
                            onChange={(e) => setProjectTitleDraft(e.target.value)}
                            onClick={(e) => e.stopPropagation()}
                            onKeyDown={(e) => {
                              e.stopPropagation();
                              if (e.key === 'Enter') {
                                e.preventDefault();
                                saveProjectTitle(project.job_id);
                              } else if (e.key === 'Escape') {
                                e.preventDefault();
                                cancelEditProjectTitle();
                              }
                            }}
                            className="bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg py-2 px-3 text-sm"
                            autoFocus
                          />
                          <button
                            type="button"
                            onClick={(e) => {
                              e.stopPropagation();
                              saveProjectTitle(project.job_id);
                            }}
                            className="px-2 py-1 text-xs rounded-lg border border-emerald-300 bg-emerald-100 text-emerald-700 dark:bg-emerald-900/20 dark:border-emerald-700 dark:text-emerald-300"
                          >
                            Guardar
                          </button>
                          <button
                            type="button"
                            onClick={(e) => {
                              e.stopPropagation();
                              cancelEditProjectTitle();
                            }}
                            className="px-2 py-1 text-xs rounded-lg border border-slate-300 text-slate-600 dark:border-slate-600 dark:text-slate-300"
                          >
                            Cancelar
                          </button>
                        </div>
                      ) : (
                        <div className="flex items-center gap-2 min-w-0">
                          <h3 className="text-xl md:text-2xl font-bold text-slate-900 dark:text-white truncate">{project.title || 'Proyecto'}</h3>
                          <button
                            type="button"
                            onClick={(e) => {
                              e.stopPropagation();
                              beginEditProjectTitle(project);
                            }}
                            className="p-1 rounded-md text-slate-400 hover:text-primary hover:bg-primary/10 transition-colors shrink-0"
                            title="Editar título"
                          >
                            <Pencil size={14} />
                          </button>
                        </div>
                      )}
                      <div className="text-sm text-slate-500 dark:text-slate-400 space-y-0.5 mt-1">
                        <p>{`Creado: ${formatProjectDate(project.created_at)}`}</p>
                        <p>{`Expira: ${formatProjectDate(project.expires_at)}`}</p>
                      </div>
                    </div>
                  </div>

                  <div className="col-span-1 md:col-span-2">
                    <span
                      className={`inline-flex max-w-full items-center gap-1.5 rounded-full px-3 py-1 text-xs font-medium ${projectSourceBadgeClass(project.source_kind)}`}
                      title={project.source_label || 'Archivo local'}
                    >
                      <span className="shrink-0">
                        {project.source_kind === 'youtube' ? <Youtube size={13} /> : project.source_kind === 'url' ? <Link2 size={13} /> : <Upload size={13} />}
                      </span>
                      <span className="truncate">{project.source_label || 'Archivo local'}</span>
                    </span>
                  </div>

                  <div className="col-span-1 md:col-span-3">
                    <div className="flex flex-col">
                      <span className="font-semibold text-slate-900 dark:text-white text-sm">{project.video_type || 'Topic-clips'}</span>
                      <span className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
                        {project.clip_count_actual
                          ? `Número de clips: ${project.clip_count_actual}`
                          : project.status === 'processing'
                            ? 'Procesando...'
                            : project.status === 'error'
                              ? 'Error en procesamiento'
                              : `Objetivo: ${project.clip_count_target || '-'} clips`}
                      </span>
                    </div>
                  </div>

                  <div className="col-span-1 md:col-span-1">
                    <span className="font-medium text-slate-700 dark:text-slate-300 text-sm">{outputModeLabel(project.ratio)}</span>
                  </div>

                  <div className="col-span-1 md:col-span-1 flex items-center justify-end gap-1">
                    <span className={`text-[11px] px-2 py-0.5 rounded-full border ${project.status === 'complete'
                      ? 'bg-emerald-100 text-emerald-700 border-emerald-200 dark:bg-emerald-900/20 dark:text-emerald-300 dark:border-emerald-800'
                      : project.status === 'error'
                        ? 'bg-red-100 text-red-700 border-red-200 dark:bg-red-900/20 dark:text-red-300 dark:border-red-800'
                        : 'bg-amber-100 text-amber-700 border-amber-200 dark:bg-amber-900/20 dark:text-amber-300 dark:border-amber-800'
                      }`}>
                      {projectStatusLabel(project.status)}
                    </span>
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation();
                        toggleProjectFavorite(project.job_id);
                      }}
                      className={`p-2 rounded-full transition-colors ${project.favorite
                        ? 'text-pink-500 bg-pink-50 dark:bg-pink-900/20'
                        : 'text-slate-400 hover:text-pink-500 hover:bg-slate-100 dark:hover:bg-slate-800'
                        }`}
                      title={project.favorite ? 'Quitar de favoritos' : 'Agregar a favoritos'}
                    >
                      <Heart size={16} fill={project.favorite ? 'currentColor' : 'none'} />
                    </button>
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation();
                        removeProject(project.job_id);
                      }}
                      className="p-2 rounded-full text-slate-400 hover:text-red-600 dark:hover:text-red-400 hover:bg-red-50 dark:hover:bg-red-900/20 transition-colors"
                      title="Eliminar proyecto"
                    >
                      <Trash2 size={16} />
                    </button>
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation();
                        openSavedProject(project);
                        setProjectsViewMode('detail');
                      }}
                      className="inline-flex items-center gap-1.5 rounded-lg bg-primary px-3 py-2 text-xs font-semibold text-white transition-colors hover:bg-slate-800"
                      title="Abrir proyecto"
                    >
                      Abrir
                      <ChevronRight size={14} />
                    </button>
                  </div>
                </div>
                {project.status === 'processing' && (
                  <div className="absolute left-0 right-0 bottom-0 h-1 bg-violet-100 dark:bg-slate-700">
                    <div className="h-full w-1/3 bg-primary animate-pulse" />
                  </div>
                )}
              </div>
            ))}
          </div>
          <button
            type="button"
            onClick={() => {
              setActiveTab('home');
              setProjectsViewMode('list');
              window.scrollTo({ top: 0, behavior: 'smooth' });
            }}
            className="w-full mt-6 flex items-center justify-center gap-2 py-4 border-2 border-dashed border-slate-200 dark:border-slate-700 rounded-2xl text-slate-500 dark:text-slate-400 hover:border-primary hover:text-primary transition-all group"
          >
            <div className="w-10 h-10 rounded-full bg-slate-100 dark:bg-slate-800 flex items-center justify-center group-hover:bg-primary/10 transition-colors">
              <Upload size={20} />
            </div>
            <span className="font-semibold text-lg">Crear nuevo proyecto</span>
          </button>
        </>
      ) : (
        <div className="flex flex-col items-center justify-center py-20 px-6 text-center">
          <div className="w-20 h-20 rounded-full bg-slate-100 dark:bg-slate-800 flex items-center justify-center mb-6">
            <Activity size={40} className="text-slate-400" />
          </div>
          <h3 className="text-2xl font-bold text-slate-900 dark:text-white mb-2">No hay proyectos todavía</h3>
          <p className="text-slate-500 dark:text-slate-400 max-w-sm mb-8">
            Cuando proceses videos, aparecerán aquí para que puedas acceder a ellos en cualquier momento.
          </p>
          <button
            onClick={() => setActiveTab('home')}
            className="inline-flex items-center gap-2 px-6 py-3 rounded-xl bg-primary text-white font-bold hover:bg-slate-800 transition-colors shadow-lg shadow-primary/20"
          >
            Empezar ahora
          </button>
        </div>
      )}
    </div>
  );
};

export default ProjectsView;
