import os as _0
import cv2 as _1

_2 = 0.5625
_3 = None
_4 = None
_5 = None


def _6():
    global _3
    if _3 is None:
        try:
            from ultralytics import YOLO as _7
            _3 = _7("yolov8n.pt")
        except Exception:
            _3 = False
    return _3


def _8():
    global _4, _5
    if _5 is None:
        try:
            import mediapipe as _9
            _4 = _9.solutions.face_detection
            _5 = _4.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        except Exception:
            _5 = False
    return _5


class _10:
    def __init__(self, a, b):
        self.a = a
        self.b = b if b else 1e-6

    def c(self):
        return self.a

    def d(self):
        return self.a / self.b


def x1(a, b=0, c=0):
    d = []
    e = 30.0
    f = 1
    g = _1.VideoCapture(a)
    if g.isOpened():
        e = float(g.get(_1.CAP_PROP_FPS) or 30.0)
        f = int(g.get(_1.CAP_PROP_FRAME_COUNT) or 1)
        g.release()
    if not d:
        d = [(_10(0, e), _10(f, e))]
    return d, e


class x2:
    def __init__(self, a, b, c, d):
        self.e = a
        self.f = b
        self.g = c
        self.h = d
        self.i = c / 2
        self.j = c / 2
        self.k = d
        self.l = int(self.k * _2)
        if self.l > c:
            self.l = c
            self.k = int(self.l / _2)
        self.m = self.l * 0.25

    def n(self, a):
        if a:
            b, c, d, e = a
            self.j = b + d / 2

    def o(self, a=False):
        if a:
            self.i = self.j
        else:
            b = self.j - self.i
            if abs(b) > self.m:
                c = 1 if b > 0 else -1
                d = 15.0 if abs(b) > self.l * 0.5 else 3.0
                self.i += c * d
                e = self.j - self.i
                if (c == 1 and e < 0) or (c == -1 and e > 0):
                    self.i = self.j
        f = self.l / 2
        if self.i - f < 0:
            self.i = f
        if self.i + f > self.g:
            self.i = self.g - f
        g = max(0, int(self.i - f))
        h = min(self.g, int(self.i + f))
        return g, 0, h, self.h


class x3:
    def __init__(self, a=15, b=30):
        self.p = None
        self.q = {}
        self.r = {}
        self.s = 0
        self.t = a
        self.u = b
        self.v = -1000
        self.w = 0
        self.z = []

    def aa(self, a, b, c):
        d = []
        for e in a:
            f, g, h, i = e["box"]
            j = f + h / 2
            k = -1
            l = c * 0.15
            for m in self.z:
                if b - m["f"] > 30:
                    continue
                n = abs(j - m["c"])
                if n < l:
                    l = n
                    k = m["i"]
            if k == -1:
                k = self.w
                self.w += 1
            self.z = [m for m in self.z if m["i"] != k]
            self.z.append({"i": k, "c": j, "f": b})
            d.append({"i": k, "box": e["box"], "score": e["score"]})

        for e in list(self.q.keys()):
            self.q[e] *= 0.85
            if self.q[e] < 0.1:
                del self.q[e]

        for e in d:
            f = e["i"]
            g = e["score"] / (c * c * 0.05)
            self.q[f] = self.q.get(f, 0) + g

        if not d:
            return None

        h = None
        i = -1
        for e in d:
            f = e["i"]
            g = self.q.get(f, 0)
            if f == self.p:
                g *= 3.0
            if g > i:
                i = g
                h = e

        if h:
            j = h["i"]
            if j == self.p:
                self.s += 1
                return h["box"]
            if b - self.v < self.u:
                k = next((e for e in d if e["i"] == self.p), None)
                if k:
                    return k["box"]
            self.p = j
            self.v = b
            self.s = 0
            return h["box"]
        return None


def x4(a):
    try:
        b, c, _ = a.shape
        d = _1.cvtColor(a, _1.COLOR_BGR2RGB)
        e = _8()
        if not e:
            return []
        f = e.process(d)
        g = []
        if not f.detections:
            return []
        for h in f.detections:
            i = h.location_data.relative_bounding_box
            j = int(i.xmin * c)
            k = int(i.ymin * b)
            l = int(i.width * c)
            m = int(i.height * b)
            g.append({"box": [j, k, l, m], "score": l * m})
        return g
    except Exception:
        return []


def x5(a):
    try:
        b = _6()
        if not b:
            return None
        c = b(a, verbose=False, classes=[0])
        if not c:
            return None
        d = None
        e = 0
        for f in c:
            for g in f.boxes:
                h, i, j, k = [int(v) for v in g.xyxy[0]]
                l = j - h
                m = k - i
                n = l * m
                if n > e:
                    e = n
                    d = [h, i, l, int(m * 0.4)]
        return d
    except Exception:
        return None


def x6(a, b):
    c = _1.VideoCapture(a)
    d = []
    if not c.isOpened():
        return ["A"] * len(b)
    for e, f in b:
        g = [
            e.c() + 5,
            int((e.c() + f.c()) / 2),
            f.c() - 5,
        ]
        h = []
        for i in g:
            c.set(_1.CAP_PROP_POS_FRAMES, i)
            j, k = c.read()
            if not j:
                continue
            l = x4(k)
            h.append(len(l))
        m = 0 if not h else sum(h) / len(h)
        d.append("B" if 0.5 <= m <= 1.2 else "A")
    c.release()
    return d


def x7(a):
    return False


def x8(a, b):
    return False