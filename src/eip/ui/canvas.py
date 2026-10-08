import math

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QGraphicsPixmapItem, QGraphicsRectItem, QGraphicsScene, QGraphicsView

from eip.domain import ROI


class ImageCanvas(QGraphicsView):
    roi_changed = Signal(object)
    pixel_moved = Signal(int, int)
    view_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.pixmap = QGraphicsPixmapItem()
        self.scene().addItem(self.pixmap)
        self.roi_item = QGraphicsRectItem()
        pen = QPen(QColor("#44c3e2"), 1.5)
        pen.setCosmetic(True)
        self.roi_item.setPen(pen)
        self.roi_item.setBrush(QColor(68, 195, 226, 25))
        self.roi_item.setZValue(2)
        self.scene().addItem(self.roi_item)
        self.roi_item.hide()
        self.setBackgroundBrush(QColor("#12171d"))
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setMouseTracking(True)
        self.roi_mode = False
        self.anchor = None
        self.image_shape = None
        self._syncing = False
        self.horizontalScrollBar().valueChanged.connect(self._emit_view)
        self.verticalScrollBar().valueChanged.connect(self._emit_view)

    def set_image(self, rgb, origin=(0, 0), base_shape=None, fit=False):
        self._syncing = True
        h, w, _ = rgb.shape
        image = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format.Format_RGB888).copy()
        self.pixmap.setPixmap(QPixmap.fromImage(image))
        self.pixmap.setPos(*origin)
        self.image_shape = base_shape or (h, w)
        self.setSceneRect(QRectF(0, 0, self.image_shape[1], self.image_shape[0]))
        self._syncing = False
        if fit:
            self.fit_image()

    def clear_image(self):
        self.pixmap.setPixmap(QPixmap())
        self.image_shape = None
        self.roi_item.hide()

    def fit_image(self):
        if self.image_shape:
            self._syncing = True
            self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
            self._syncing = False
            self._emit_view()

    def set_roi(self, roi: ROI | None):
        if roi:
            self.roi_item.setRect(roi.x, roi.y, roi.width, roi.height)
            self.roi_item.show()
        else:
            self.roi_item.hide()

    def set_roi_mode(self, enabled: bool):
        self.roi_mode = enabled
        self.setDragMode(QGraphicsView.DragMode.NoDrag if enabled else QGraphicsView.DragMode.ScrollHandDrag)
        self.setCursor(Qt.CursorShape.CrossCursor if enabled else Qt.CursorShape.ArrowCursor)

    def _point(self, event):
        point = self.mapToScene(event.position().toPoint())
        if self.image_shape:
            h, w = self.image_shape[:2]
            return QPointF(max(0, min(w, point.x())), max(0, min(h, point.y())))
        return point

    def mousePressEvent(self, event):
        if self.roi_mode and self.image_shape and event.button() == Qt.MouseButton.LeftButton:
            self.anchor = self._point(event)
            self.roi_item.setRect(QRectF(self.anchor, self.anchor))
            self.roi_item.show()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        point = self.mapToScene(event.position().toPoint())
        self.pixel_moved.emit(math.floor(point.x()), math.floor(point.y()))
        if self.anchor is not None:
            self.roi_item.setRect(QRectF(self.anchor, self._point(event)).normalized())
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.anchor is not None:
            rectangle = QRectF(self.anchor, self._point(event)).normalized()
            x0, y0 = math.floor(rectangle.left()), math.floor(rectangle.top())
            x1, y1 = math.ceil(rectangle.right()), math.ceil(rectangle.bottom())
            self.anchor = None
            if x1 > x0 and y1 > y0:
                roi = ROI(x0, y0, x1 - x0, y1 - y0)
                self.set_roi(roi)
                self.roi_changed.emit(roi)
            else:
                self.set_roi(None)
                self.roi_changed.emit(None)
            return
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        factor = 1.2 if event.angleDelta().y() > 0 else 1 / 1.2
        scale = self.transform().m11() * factor
        if 0.005 <= scale <= 100:
            self.scale(factor, factor)
            self._emit_view()
        event.accept()

    def _emit_view(self):
        if not self._syncing:
            self.view_changed.emit()

    def sync_from(self, other):
        if self.image_shape is None or other.image_shape is None:
            return
        self._syncing = True
        self.setTransform(other.transform())
        self.centerOn(other.mapToScene(other.viewport().rect().center()))
        self._syncing = False
