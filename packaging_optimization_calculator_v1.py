import sys
import qrcode
import os
import random
import copy
import numpy as np
from fpdf import FPDF
from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QFormLayout, QLineEdit, QLabel,
                               QSpinBox, QComboBox, QPushButton, QTextEdit, QFileDialog, QCheckBox,
                               QMessageBox, QTabWidget, QProgressBar, QGraphicsDropShadowEffect, QGraphicsOpacityEffect)
from PySide6.QtCore import Qt, QTimer, QEasingCurve, QPropertyAnimation
from PySide6.QtGui import QFont, QDoubleValidator
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

#############################################
# Class Container – represents a container and its packing methods
#############################################
class Container:
    def __init__(self, max_weight, max_volume, container_type, container_id, dimensions):
        self.max_weight = max_weight
        self.max_volume = max_volume
        # container_type will be used later to display the container type info
        self.container_type = container_type  
        self.container_id = container_id
        self.dimensions = dimensions  # (height, width, length) in cm
        self.current_weight = 0
        self.current_length_usage = 0    # for non-stackable packing
        self.current_layer_height = 0      # for stackable packing
        self.pallets = []                # list of dictionaries with pallet information

    def can_pack(self, pallet):
        return (self.current_weight + pallet['weight'] <= self.max_weight) and \
               (self.calculate_used_volume() + pallet['volume'] <= self.max_volume)

    def calculate_used_volume(self):
        return sum(p['volume'] for p in self.pallets)

    def pack(self, pallet):
        # For non-stackable packing, pallets are arranged along the container length.
        if not pallet.get('stackable', False):
            if self.current_length_usage + pallet['length'] <= self.dimensions[2] and self.can_pack(pallet):
                pallet['x'] = 0
                pallet['y'] = 0
                pallet['z'] = self.current_length_usage
                self.current_length_usage += pallet['length']
                self.pallets.append(pallet)
                self.current_weight += pallet['weight']
                return True
            return False
        else:
            # For stackable pallets, use vertical stacking.
            if self.current_layer_height + pallet['height'] <= self.dimensions[0] and self.can_pack(pallet):
                pallet['x'] = 0
                pallet['y'] = self.current_layer_height
                pallet['z'] = 0
                self.current_layer_height += pallet['height']
                self.pallets.append(pallet)
                self.current_weight += pallet['weight']
                return True
            return False

#############################################
# Class SpaceOptimizer – space optimization (B&B + fallback)
#############################################
class SpaceOptimizer:
    def optimize(self, container_dimensions, container_max_weight, container_max_volume, pallets, container_type_str):
        if len(pallets) <= 10:
            best = [float('inf')]
            solution = self._bb_recursive(pallets, [], best, container_dimensions, container_max_weight, container_max_volume, 1, container_type_str)
            if solution is not None:
                for idx, cont in enumerate(solution, start=1):
                    cont.container_id = "Container {}".format(idx)
                return solution
        return self._greedy_optimize(container_dimensions, container_max_weight, container_max_volume, pallets, container_type_str)

    def _bb_recursive(self, pallets, containers, best, container_dimensions, container_max_weight, container_max_volume, next_container_id, container_type_str):
        if not pallets:
            return copy.deepcopy(containers)
        if len(containers) >= best[0]:
            return None

        current_best = None
        first = pallets[0]
        remaining = pallets[1:]
        for cont in containers:
            old_pallets = cont.pallets.copy()
            old_weight = cont.current_weight
            if first.get('stackable', False):
                old_layer = cont.current_layer_height
            else:
                old_length = cont.current_length_usage

            if cont.pack(first):
                sol = self._bb_recursive(remaining, containers, best, container_dimensions, container_max_weight, container_max_volume, next_container_id, container_type_str)
                if sol is not None and len(sol) < best[0]:
                    best[0] = len(sol)
                    current_best = sol
                cont.pallets = old_pallets
                cont.current_weight = old_weight
                if first.get('stackable', False):
                    cont.current_layer_height = old_layer
                else:
                    cont.current_length_usage = old_length

        new_container = Container(max_weight=container_max_weight, max_volume=container_max_volume,
                                  container_type=container_type_str, container_id="Container {}".format(next_container_id),
                                  dimensions=container_dimensions)
        if new_container.pack(first):
            new_containers = containers[:] + [new_container]
            sol = self._bb_recursive(remaining, new_containers, best, container_dimensions, container_max_weight, container_max_volume, next_container_id + 1, container_type_str)
            if sol is not None and (current_best is None or len(sol) < len(current_best)):
                current_best = sol
                best[0] = len(sol)
        return current_best

    def _greedy_optimize(self, container_dimensions, container_max_weight, container_max_volume, pallets, container_type_str):
        containers = []
        container_id = 1
        container = Container(max_weight=container_max_weight, max_volume=container_max_volume,
                              container_type=container_type_str, container_id="Container {}".format(container_id),
                              dimensions=container_dimensions)
        containers.append(container)
        for pallet in pallets:
            if not container.pack(pallet):
                container_id += 1
                container = Container(max_weight=container_max_weight, max_volume=container_max_volume,
                                      container_type=container_type_str, container_id="Container {}".format(container_id),
                                      dimensions=container_dimensions)
                containers.append(container)
                container.pack(pallet)
        return containers

#############################################
# Class LoadStabilityOptimizer – load stability and cargo safety optimization (BFD, LAAF)
#############################################
class LoadStabilityOptimizer:
    def __init__(self, strategy):
        self.strategy = strategy

    def optimize(self, container_dimensions, container_max_weight, container_max_volume, pallets, container_type_str):
        if self.strategy == "Load Stability":
            return self._bfd_optimize(container_dimensions, container_max_weight, container_max_volume, pallets, container_type_str)
        elif self.strategy == "Cargo Safety":
            return self._laaf_optimize(container_dimensions, container_max_weight, container_max_volume, pallets, container_type_str)
        else:
            return self._bfd_optimize(container_dimensions, container_max_weight, container_max_volume, pallets, container_type_str)

    def _bfd_optimize(self, container_dimensions, container_max_weight, container_max_volume, pallets, container_type_str):
        sorted_pallets = sorted(pallets, key=lambda p: p['volume'], reverse=True)
        containers = []
        container_id = 1
        for pallet in sorted_pallets:
            best_fit = None
            best_slack = None
            for cont in containers:
                if pallet.get('stackable', False):
                    remaining = container_dimensions[0] - cont.current_layer_height
                    dim = pallet['height']
                else:
                    remaining = container_dimensions[2] - cont.current_length_usage
                    dim = pallet['length']
                if cont.can_pack(pallet) and remaining >= dim:
                    slack = remaining - dim
                    if best_slack is None or slack < best_slack:
                        best_slack = slack
                        best_fit = cont
            if best_fit is not None:
                best_fit.pack(pallet)
            else:
                new_container = Container(max_weight=container_max_weight, max_volume=container_max_volume,
                                          container_type=container_type_str, container_id="Container {}".format(container_id),
                                          dimensions=container_dimensions)
                new_container.pack(pallet)
                containers.append(new_container)
                container_id += 1
        return containers

    def _laaf_optimize(self, container_dimensions, container_max_weight, container_max_volume, pallets, container_type_str):
        containers = []
        container_id = 1
        for i, pallet in enumerate(pallets):
            placed = False
            remaining_pallets = pallets[i+1:]
            if pallet.get('stackable', False):
                avg_remaining = (sum(p['height'] for p in remaining_pallets) / len(remaining_pallets)) if remaining_pallets else pallet['height']
            else:
                avg_remaining = (sum(p['length'] for p in remaining_pallets) / len(remaining_pallets)) if remaining_pallets else pallet['length']
            for cont in containers:
                if pallet.get('stackable', False):
                    remaining_space = container_dimensions[0] - cont.current_layer_height
                else:
                    remaining_space = container_dimensions[2] - cont.current_length_usage
                if remaining_space < avg_remaining:
                    continue
                if cont.can_pack(pallet) and cont.pack(pallet):
                    placed = True
                    break
            if not placed:
                new_container = Container(max_weight=container_max_weight, max_volume=container_max_volume,
                                          container_type=container_type_str, container_id="Container {}".format(container_id),
                                          dimensions=container_dimensions)
                new_container.pack(pallet)
                containers.append(new_container)
                container_id += 1
        return containers

#############################################
# Class PackagingController – selects the optimization strategy
#############################################
class PackagingController:
    def __init__(self, strategy):
        self.strategy = strategy

    def pack(self, container_dimensions, container_max_weight, container_max_volume, pallets, container_type_str):
        if self.strategy == 'Space Optimization':
            optimizer = SpaceOptimizer()
        elif self.strategy in ['Load Stability', 'Cargo Safety']:
            optimizer = LoadStabilityOptimizer(self.strategy)
        else:
            optimizer = SpaceOptimizer()
        return optimizer.optimize(container_dimensions, container_max_weight, container_max_volume, pallets, container_type_str)

#############################################
# Helper functions for drawing
#############################################
def draw_wireframe_container(ax, origin, size, color='blue'):
    x0, y0, z0 = origin
    dx, dy, dz = size
    vertices = np.array([
        [x0,     y0,     z0],
        [x0+dx,  y0,     z0],
        [x0+dx,  y0+dy,  z0],
        [x0,     y0+dy,  z0],
        [x0,     y0,     z0+dz],
        [x0+dx,  y0,     z0+dz],
        [x0+dx,  y0+dy,  z0+dz],
        [x0,     y0+dy,  z0+dz]
    ])
    edges = [
        [0,1], [1,2], [2,3], [3,0],
        [4,5], [5,6], [6,7], [7,4],
        [0,4], [1,5], [2,6], [3,7]
    ]
    for e in edges:
        p1, p2 = vertices[e[0]], vertices[e[1]]
        ax.plot3D(*zip(p1, p2), color=color)

def draw_colored_box_with_label(ax, origin, size, label="Box", box_color=None):
    x0, y0, z0 = origin
    dx, dy, dz = size
    if box_color is None:
        box_color = np.random.rand(3,)
    vertices = np.array([
        [x0,     y0,     z0],
        [x0+dx,  y0,     z0],
        [x0+dx,  y0+dy,  z0],
        [x0,     y0+dy,  z0],
        [x0,     y0,     z0+dz],
        [x0+dx,  y0,     z0+dz],
        [x0+dx,  y0+dy,  z0+dz],
        [x0,     y0+dy,  z0+dz]
    ])
    faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]],
        [vertices[4], vertices[5], vertices[6], vertices[7]],
        [vertices[0], vertices[1], vertices[5], vertices[4]],
        [vertices[2], vertices[3], vertices[7], vertices[6]],
        [vertices[1], vertices[2], vertices[6], vertices[5]],
        [vertices[4], vertices[7], vertices[3], vertices[0]]
    ]
    poly3d = Poly3DCollection(faces, facecolors=box_color, edgecolors='k', alpha=0.8)
    ax.add_collection3d(poly3d)
    cx = x0 + dx/2
    cy = y0 + dy/2
    cz = z0 + dz
    ax.text(cx, cy, cz, label, color='white', fontsize=9, ha='center', va='bottom', zdir='z')

#############################################
# Class PackagingApp – user interface and visualization
#############################################
class PackagingApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Optimization Calculator')
        self.setGeometry(100, 100, 900, 900)
        self.containers = []
        self.init_ui()
        self.apply_styles()

    def init_ui(self):
        self.main_layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        self.config_tab = QWidget()
        self.results_tab = QWidget()
        self.visual_tab = QWidget()
        
        self.tabs.addTab(self.config_tab, "Configuration")
        self.tabs.addTab(self.results_tab, "Results")
        self.tabs.addTab(self.visual_tab, "Visualization")
        
        self.init_config_tab()
        self.init_results_tab()
        self.init_visual_tab()
        
        self.main_layout.addWidget(self.tabs)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.main_layout.addWidget(self.progress_bar)
        
        # Notification area
        self.notification_label = QLabel("", self)
        self.notification_label.setAlignment(Qt.AlignCenter)
        self.notification_label.setVisible(False)
        self.main_layout.addWidget(self.notification_label)

    def init_config_tab(self):
        layout = QVBoxLayout(self.config_tab)
        layout.setContentsMargins(20, 20, 20, 20)
        
        title = QLabel('<b>Optimization Calculator</b>')
        title.setFont(QFont('Segoe UI', 28, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        form_layout = QFormLayout()
        self.pallet_name_input = QLineEdit()
        self.pallet_name_input.setPlaceholderText("e.g. Pallet A")
        self.length_input = QLineEdit()
        self.length_input.setPlaceholderText("e.g. 120")
        self.width_input = QLineEdit()
        self.width_input.setPlaceholderText("e.g. 80")
        self.height_input = QLineEdit()
        self.height_input.setPlaceholderText("e.g. 100")
        self.weight_input = QLineEdit()
        self.weight_input.setPlaceholderText("e.g. 150")
        
        double_validator = QDoubleValidator(0.0, 10000.0, 2)
        self.length_input.setValidator(double_validator)
        self.width_input.setValidator(double_validator)
        self.height_input.setValidator(double_validator)
        self.weight_input.setValidator(double_validator)
        
        self.quantity_input = QSpinBox()
        self.quantity_input.setMinimum(1)
        self.quantity_input.setMaximum(1000)
        self.strategy_input = QComboBox()
        self.strategy_input.addItems(['Space Optimization', 'Load Stability', 'Cargo Safety'])
        self.stackable_checkbox = QCheckBox("Stackable")
        self.container_type_input = QComboBox()
        self.container_type_input.addItems(["20 ft", "40 ft"])

        form_layout.addRow(QLabel('Pallet Name:'), self.pallet_name_input)
        form_layout.addRow(QLabel('Length (cm):'), self.length_input)
        form_layout.addRow(QLabel('Width (cm):'), self.width_input)
        form_layout.addRow(QLabel('Height (cm):'), self.height_input)
        form_layout.addRow(QLabel('Weight (kg):'), self.weight_input)
        form_layout.addRow(QLabel('Quantity:'), self.quantity_input)
        form_layout.addRow(QLabel('Strategy:'), self.strategy_input)
        form_layout.addRow(QLabel('Stackable:'), self.stackable_checkbox)
        form_layout.addRow(QLabel('Container Type:'), self.container_type_input)
        
        layout.addLayout(form_layout)
        
        self.submit_button = QPushButton('Calculate')
        self.submit_button.clicked.connect(self.on_submit)
        self.add_shadow(self.submit_button)
        self.add_button_animation(self.submit_button)
        layout.addWidget(self.submit_button)

        self.label_button = QPushButton('Generate Labels')
        self.label_button.clicked.connect(self.generate_labels)
        self.add_shadow(self.label_button)
        self.add_button_animation(self.label_button)
        layout.addWidget(self.label_button)

    def init_results_tab(self):
        layout = QVBoxLayout(self.results_tab)
        layout.setContentsMargins(20, 20, 20, 20)
        self.results_output = QTextEdit()
        self.results_output.setReadOnly(True)
        layout.addWidget(self.results_output)
    
    def init_visual_tab(self):
        layout = QVBoxLayout(self.visual_tab)
        layout.setContentsMargins(20, 20, 20, 20)
        self.visualize_button = QPushButton('View Visualization')
        self.visualize_button.clicked.connect(self.visualize_packing)
        self.add_shadow(self.visualize_button)
        self.add_button_animation(self.visualize_button)
        layout.addWidget(self.visualize_button)
    
    def add_shadow(self, widget):
        effect = QGraphicsDropShadowEffect()
        effect.setBlurRadius(15)
        effect.setOffset(0, 3)
        widget.setGraphicsEffect(effect)
    
    def add_button_animation(self, button):
        effect = QGraphicsOpacityEffect(button)
        button.setGraphicsEffect(effect)
        animation = QPropertyAnimation(effect, b"opacity")
        animation.setDuration(200)
        animation.setStartValue(1.0)
        animation.setEndValue(0.7)
        animation.setEasingCurve(QEasingCurve.InOutQuad)
        button.clicked.connect(lambda: self.run_button_animation(animation, effect))
    
    def run_button_animation(self, animation, effect):
        animation.setDirection(QPropertyAnimation.Forward)
        animation.start()
        QTimer.singleShot(200, lambda: animation.setDirection(QPropertyAnimation.Backward))
        QTimer.singleShot(200, lambda: animation.start())

    def apply_styles(self):
        self.setStyleSheet("""
            QWidget {
                background-color: #1E1E2F;
                color: #ECEFF1;
                font-family: 'Segoe UI', sans-serif;
            }
            QTabWidget::pane {
                border: 1px solid #3A3A4A;
                border-radius: 4px;
                margin: 2px;
            }
            QTabBar::tab {
                background: #3A3A4A;
                padding: 10px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background: #FF4081;
                font-weight: bold;
            }
            QPushButton {
                background-color: #FF4081;
                border: none;
                border-radius: 4px;
                padding: 10px 20px;
                font-size: 16px;
                font-weight: bold;
                color: #FFFFFF;
            }
            QPushButton:hover {
                background-color: #E73370;
            }
            QPushButton:pressed {
                background-color: #CC295C;
            }
            QLineEdit, QComboBox, QSpinBox {
                background-color: #ECEFF1;
                color: #1E1E2F;
                border: 1px solid #B0BEC5;
                border-radius: 4px;
                padding: 5px;
            }
            QTextEdit {
                background-color: #ECEFF1;
                color: #1E1E2F;
                border: 1px solid #B0BEC5;
                border-radius: 4px;
                padding: 5px;
            }
            QProgressBar {
                background-color: #B0BEC5;
                border: none;
                border-radius: 4px;
                text-align: center;
                color: #1E1E2F;
            }
            QProgressBar::chunk {
                background-color: #FF4081;
                border-radius: 4px;
            }
        """)

    def show_notification(self, message, duration=3000):
        self.notification_label.setText(message)
        self.notification_label.setStyleSheet("""
            background-color: #FF4081;
            color: white;
            border-radius: 4px;
            padding: 10px;
            font-weight: bold;
        """)
        self.notification_label.setVisible(True)
        opacity_effect = QGraphicsOpacityEffect(self.notification_label)
        self.notification_label.setGraphicsEffect(opacity_effect)
        animation = QPropertyAnimation(opacity_effect, b"opacity")
        animation.setDuration(1000)
        animation.setStartValue(1.0)
        animation.setEndValue(0.0)
        animation.setEasingCurve(QEasingCurve.InOutQuad)
        QTimer.singleShot(duration, lambda: animation.start())
        QTimer.singleShot(duration + 1000, lambda: self.notification_label.setVisible(False))

    def on_submit(self):
        self.containers = []
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        
        try:
            pallet_name = self.pallet_name_input.text().strip() or "Pallet"
            length = float(self.length_input.text())
            width = float(self.width_input.text())
            height = float(self.height_input.text())
            weight = float(self.weight_input.text())
            quantity = self.quantity_input.value()
            strategy = self.strategy_input.currentText()
            stackable = self.stackable_checkbox.isChecked()
            container_type = self.container_type_input.currentText()
        except Exception as e:
            self.progress_bar.setVisible(False)
            QMessageBox.critical(self, "Data Error", "Data conversion error: " + str(e))
            return

        pallet_volume = length * width * height
        pallets = []
        for i in range(quantity):
            pallet = {
                'name': pallet_name,
                'length': length,
                'width': width,
                'height': height,
                'weight': weight,
                'volume': pallet_volume,
                'stackable': stackable
            }
            pallets.append(pallet)

        if container_type == "20 ft":
            container_dimensions = (235, 240, 600)  # in cm
            container_size_str = "20 ft container"
        else:
            container_dimensions = (235, 240, 1200)  # in cm
            container_size_str = "40 ft container"
        
        # --- Heuristic improvement ---
        # If pallets are marked as stackable but the total length fits within the container,
        # treat them as non-stackable for optimal packing.
        if stackable and (quantity * length) <= container_dimensions[2]:
            stackable = False
            for pallet in pallets:
                pallet['stackable'] = False
        # --------------------------------

        container_max_weight = 10000
        container_max_volume = container_dimensions[0] * container_dimensions[1] * container_dimensions[2]

        controller = PackagingController(strategy)
        QTimer.singleShot(100, lambda: self.process_packing(controller, container_dimensions, container_max_weight, container_max_volume, pallets, container_size_str))

    def process_packing(self, controller, container_dimensions, container_max_weight, container_max_volume, pallets, container_size_str):
        # If pallets are non-stackable and the total length fits in the container,
        # pack all pallets into one container directly.
        if not pallets[0]['stackable'] and sum(p['length'] for p in pallets) <= container_dimensions[2]:
            new_container = Container(max_weight=container_max_weight, max_volume=container_max_volume,
                                      container_type=container_size_str, container_id="Container 1",
                                      dimensions=container_dimensions)
            for pallet in pallets:
                new_container.pack(pallet)
            self.containers = [new_container]
        else:
            self.containers = controller.pack(container_dimensions, container_max_weight, container_max_volume, pallets, container_size_str)
        
        # Remove any empty containers (if present)
        self.containers = [c for c in self.containers if c.pallets]

        results_str = "Strategy: {}\n".format(self.strategy_input.currentText())
        results_str += "Packed {} pallet(s) into {} container(s).\n\n".format(len(pallets), len(self.containers))
        for cont in self.containers:
            # Convert container dimensions and volume from cm to m
            dim_m = (cont.dimensions[0] / 100, cont.dimensions[1] / 100, cont.dimensions[2] / 100)
            used_volume_m3 = cont.calculate_used_volume() / 1e6
            max_volume_m3 = cont.max_volume / 1e6

            results_str += "{}:\n".format(cont.container_id)
            results_str += "  Dimensions (H x W x L): {:.2f} m x {:.2f} m x {:.2f} m\n".format(dim_m[0], dim_m[1], dim_m[2])
            results_str += "  Container Type: {}\n".format(cont.container_type)
            results_str += "  Pallets: {}\n".format(len(cont.pallets))
            results_str += "  Current weight: {} / {} kg\n".format(cont.current_weight, cont.max_weight)
            results_str += "  Used volume: {:.3f} / {:.3f} m³\n\n".format(used_volume_m3, max_volume_m3)
        self.results_output.setText(results_str)
        self.progress_bar.setVisible(False)
        self.show_notification("Packing Completed!")

    def generate_labels(self):
        if not self.containers:
            QMessageBox.warning(self, "No Data", "Calculate pallet arrangement first!")
            return

        file_path, _ = QFileDialog.getSaveFileName(self, "Save Labels", "pallet_labels.pdf", "PDF Files (*.pdf)")
        if not file_path:
            return

        pdf = FPDF()
        pdf.set_auto_page_break(auto=False)

        for container in self.containers:
            for pallet in container.pallets:
                pdf.add_page()
                pdf.set_font("Arial", size=16)
                pdf.cell(200, 10, "Pallet: {}".format(pallet['name']), ln=True, align='C')
                pdf.cell(200, 10, "Weight: {} kg".format(pallet['weight']), ln=True, align='C')
                pdf.cell(200, 10, "Dimensions: {}x{}x{} cm".format(pallet['length'], pallet['width'], pallet['height']),
                         ln=True, align='C')

                qr = qrcode.make("Pallet: {}\nWeight: {} kg".format(pallet['name'], pallet['weight']))
                qr_path = "qrcode_{}.png".format(random.randint(1000, 9999))
                qr.save(qr_path)
                pdf.image(qr_path, x=80, y=pdf.get_y() + 10, w=50, h=50)
                os.remove(qr_path)

        pdf.output(file_path)
        QMessageBox.information(self, "Labels Saved", "Labels saved at: {}".format(file_path))
        self.show_notification("Labels have been generated!")

    def visualize_packing(self):
        if not self.containers:
            QMessageBox.warning(self, "No Data", "Calculate pallet arrangement first!")
            return

        for cont in self.containers:
            container_width = cont.dimensions[1]
            container_height = cont.dimensions[0]
            container_length = cont.dimensions[2]

            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.set_title("Wireframe 3D - {}".format(cont.container_id))
            ax.set_xlabel("Width (cm)")
            ax.set_ylabel("Height (cm)")
            ax.set_zlabel("Length (cm)")

            draw_wireframe_container(ax, origin=(0, 0, 0), size=(container_width, container_height, container_length), color='blue')

            center_x = container_width / 2
            center_y = container_height / 2
            center_z = container_length / 2
            ax.text(center_x, center_y, center_z, "Pallets: {}".format(len(cont.pallets)),
                    color='red', fontsize=12, weight='bold', ha='center', va='center')

            for pallet in cont.pallets:
                if not pallet.get('stackable', False):
                    x = (container_width - pallet['width']) / 2
                    y = 0
                    z = pallet.get('z', 0)
                else:
                    x = (container_width - pallet['width']) / 2
                    y = pallet.get('y', 0)
                    z = 0

                p_width = pallet['width']
                p_height = pallet['height']
                p_length = pallet['length']

                draw_colored_box_with_label(ax, origin=(x, y, z), size=(p_width, p_height, p_length), label=pallet['name'])

            ax.set_xlim(0, container_width)
            ax.set_ylim(0, container_height)
            ax.set_zlim(0, container_length)
            plt.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PackagingApp()
    window.show()
    sys.exit(app.exec())
