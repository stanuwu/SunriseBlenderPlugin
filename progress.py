"""Runs an import as small steps: at once in background mode, or from a timer with progress in the status bar."""

import time
import traceback

import bpy


# Seconds of import work between two redraws of the window.
STEP_BUDGET = 0.1
TIMER_SECONDS = 0.02
# Errors an import reports to the user; any other error also prints its traceback.
ERRORS = (ValueError, OSError, RuntimeError, KeyError, TypeError, IndexError)
# Events that still reach the window during an import, so the user can look around.
NAVIGATION = {'MOUSEMOVE', 'INBETWEEN_MOUSEMOVE', 'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE', 'TRACKPADPAN',
              'TRACKPADZOOM', 'WINDOW_DEACTIVATE'}


def run(steps):
    """Runs every step of an import and returns its result."""
    while True:
        try:
            next(steps)
        except StopIteration as stop:
            return stop.value


def snapshot(kinds):
    """The data blocks of each kind that exist now."""
    return {kind: set(getattr(bpy.data, kind)) for kind in kinds}


def rollback(before):
    """Removes every data block of the snapshot's kinds that was added after it, in one pass."""
    added = [value for kind, previous in before.items() for value in getattr(bpy.data, kind) if value not in previous]
    if added:
        bpy.data.batch_remove(added)


class ImportSteps:
    """
    An import operator whose steps() yields (phase, done, total) and returns the import's result.
    Esc closes the steps; each import removes what it added when its steps close.
    """
    phases = ()

    def steps(self, context):
        raise NotImplementedError

    def finished(self, context, result):
        return {'FINISHED'}

    def execute(self, context):
        try:
            steps = self.steps(context)
            if bpy.app.background or context.window is None:
                return self.finished(context, run(steps))
        except ERRORS as error:
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}
        self._steps, self._started = steps, time.perf_counter()
        self._phase, self._done, self._total = self.phases[0] if self.phases else 'Starting', 0, 0
        manager = context.window_manager
        self._timer = manager.event_timer_add(TIMER_SECONDS, window=context.window)
        manager.modal_handler_add(self)
        manager.progress_begin(0, 1000)
        self._show(context)
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        if event.type == 'ESC' and event.value == 'PRESS':
            try:
                self._steps.close()
            finally:
                self._end(context)
            self.report({'WARNING'}, 'Import cancelled; nothing was added.')
            return {'CANCELLED'}
        if event.type != 'TIMER':
            return {'PASS_THROUGH'} if event.type in NAVIGATION else {'RUNNING_MODAL'}
        deadline = time.perf_counter() + STEP_BUDGET
        try:
            while time.perf_counter() < deadline:
                self._phase, self._done, self._total = next(self._steps)
        except StopIteration as stop:
            self._end(context)
            return self.finished(context, stop.value)
        except Exception as error:
            self._end(context)
            if not isinstance(error, ERRORS):
                traceback.print_exc()
            self.report({'ERROR'}, str(error) or type(error).__name__)
            return {'CANCELLED'}
        self._show(context)
        return {'RUNNING_MODAL'}

    def cancel(self, context):
        """Blender ends the operator, for example when the file closes; the import is removed."""
        try:
            self._steps.close()
        finally:
            self._end(context)

    def _fraction(self):
        index = self.phases.index(self._phase) if self._phase in self.phases else 0
        part = self._done / self._total if self._total else 0
        return min((index + part) / max(len(self.phases), 1), 1)

    def _show(self, context):
        fraction = self._fraction()
        context.window_manager.progress_update(int(fraction * 1000))
        seconds = int(time.perf_counter() - self._started)
        phase = self._phase + (f' {self._done:,} of {self._total:,}' if self._total else '')
        clock = f'{seconds // 60}:{seconds % 60:02d}'

        title = self.bl_label + ': ' + phase

        # The draw function outlives the operator if Blender ends it, so it holds only plain values.
        def draw(header, _context):
            layout = header.layout
            layout.label(text=title, icon='IMPORT')
            layout.progress(factor=fraction, type='BAR', text=f'{fraction:.0%}')
            layout.label(text=clock, icon='TIME')
            layout.label(text='Esc to cancel', icon='EVENT_ESC')

        # Setting the status text again redraws the status bar.
        context.workspace.status_text_set(draw)

    def _end(self, context):
        manager = context.window_manager
        manager.event_timer_remove(self._timer)
        manager.progress_end()
        if context.workspace is not None:
            context.workspace.status_text_set(None)
