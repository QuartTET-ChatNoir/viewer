/** Coalesce animation and input updates without cancelling a pending draw. */
export function createFrameScheduler(
  draw: (recompute: boolean) => void,
  request: (callback: FrameRequestCallback) => number = (callback) => requestAnimationFrame(callback),
  cancel: (id: number) => void = (id) => cancelAnimationFrame(id),
) {
  let pending: number | undefined;
  let dirty = false;
  return {
    schedule(recompute = true) {
      dirty ||= recompute;
      if (pending !== undefined) return;
      pending = request(() => {
        pending = undefined;
        const compute = dirty;
        dirty = false;
        draw(compute);
      });
    },
    cancel() {
      if (pending !== undefined) cancel(pending);
      pending = undefined;
      dirty = false;
    },
  };
}
