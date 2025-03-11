__version__ = "0.5.0"
minimum = "3.8"

import sys
if sys.version_info < tuple(map(int, minimum.split("."))):
    print(
        "ERROR: script", __file__, "version", __version__, "requires Python %s or later" % minimum
    )
    sys.exit(1)

import argparse
import os
import stat
import json
import re
import shutil
import tarfile
import zipfile
from urllib.request import urlopen
from pathlib import Path
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                              QHBoxLayout, QLabel, QComboBox, QLineEdit, QPushButton, 
                              QTextEdit, QFileDialog, QMessageBox)
from PySide6.QtCore import Qt, QThread, Signal
import subprocess

ARCHIVES = {}
DOMAIN = "https://developer.download.nvidia.com"

CUDA_RELEASES = {
    "CUDA 12.4.0": "12.4.0",
    "CUDA 12.4.1": "12.4.1", 
    "CUDA 12.5.0": "12.5.0",
    "CUDA 12.5.1": "12.5.1",
    "CUDA 12.6.0": "12.6.0",
    "CUDA 12.6.1": "12.6.1",
    "CUDA 12.6.2": "12.6.2",
    "CUDA 12.6.3": "12.6.3",
    "CUDA 12.8.0": "12.8.0",
    "CUDA 12.8.1": "12.8.1",
}

CUDNN_RELEASES = [
    "8.9.7",
    "9.0.0",
    "9.1.1",
    "9.2.0",
    "9.2.1",
    "9.3.0",
    "9.4.0",
    "9.5.0",
    "9.5.1",
    "9.6.0",
    "9.7.0",
    "9.7.1",
    "9.8.0",
]

PRODUCTS = {
    "CUDA Toolkit": "cuda",
    "cuDNN": "cudnn"
}

OPERATING_SYSTEMS = {
    "Windows": "windows",
    "Linux": "linux"
}

ARCHITECTURES = {
    "x86_64": "x86_64",
    "PPC64le (Linux only)": "ppc64le",
    "SBSA (Linux only)": "sbsa",
    "AARCH64 (Linux only)": "aarch64"
}

VARIANTS = {
    "CUDA 11": "cuda11",
    "CUDA 12": "cuda12"
}

COMPONENTS = {
    "All Components": None,
    "CUDA Runtime (cudart)": "cuda_cudart",
    "CXX Core Compute Libraries": "cuda_cccl",
    "CUDA Object Dump Tool": "cuda_cuobjdump",
    "CUDA Profiling Tools Interface": "cuda_cupti",
    "CUDA Demangler Tool": "cuda_cuxxfilt",
    "CUDA Demo Suite": "cuda_demo_suite",
    "CUDA Documentation": "cuda_documentation",
    "NVIDIA CUDA Compiler": "cuda_nvcc",
    "CUDA Binary Utility": "cuda_nvdisasm",
    "NVIDIA Management Library Headers": "cuda_nvml_dev",
    "CUDA Profiler": "cuda_nvprof",
    "CUDA Binary Utility": "cuda_nvprune",
    "CUDA Runtime Compilation Library": "cuda_nvrtc",
    "CUDA Tools SDK": "cuda_nvtx",
    "NVIDIA Visual Profiler": "cuda_nvvp",
    "CUDA OpenCL": "cuda_opencl",
    "CUDA Profiler API": "cuda_profiler_api",
    "CUDA Compute Sanitizer API": "cuda_sanitizer_api",
    "CUDA BLAS Library": "libcublas",
    "CUDA FFT Library": "libcufft",
    "CUDA Random Number Generation Library": "libcurand",
    "CUDA Solver Library": "libcusolver",
    "CUDA Sparse Matrix Library": "libcusparse",
    "NVIDIA Performance Primitives Library": "libnpp",
    "NVIDIA Fatbin Utilities": "libnvfatbin",
    "NVIDIA JIT Linker Library": "libnvjitlink",
    "NVIDIA JPEG Library": "libnvjpeg",
    "Nsight Compute": "nsight_compute",
    "Nsight Systems": "nsight_systems",
    "Nsight Visual Studio Edition": "nsight_vse",
    "Visual Studio Integration": "visual_studio_integration"
}

def err(msg):
    print("ERROR: " + msg)
    sys.exit(1)

def fetch_file(full_path, filename):
    download = urlopen(full_path)
    if download.status != 200:
        print("  -> Failed: " + filename)
    else:
        print(":: Fetching: " + full_path)
        with open(filename, "wb") as file:
            file.write(download.read())
            print("  -> Wrote: " + filename)

def fix_permissions(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            filename = os.path.join(root, file)
            octal = os.stat(filename)
            os.chmod(filename, octal.st_mode | stat.S_IWRITE)

def flatten_tree(src, dest, tag=None):
    if tag:
        dest = os.path.join(dest, tag)

    try:
        shutil.copytree(src, dest, symlinks=1, dirs_exist_ok=1, ignore_dangling_symlinks=1)
    except FileExistsError:
        pass
    shutil.rmtree(src)

def parse_artifact(
    parent,
    manifest,
    component,
    platform,
    retrieve=True,
    variant=None,
):
    if variant:
        full_path = parent + manifest[component][platform][variant]["relative_path"]
    else:
        full_path = parent + manifest[component][platform]["relative_path"]

    filename = os.path.basename(full_path)
    file_path = filename
    pwd = os.path.join(os.getcwd(), component, platform)

    if (
        retrieve
        and not os.path.exists(filename)
        and not os.path.exists(full_path)
        and not os.path.exists(parent + filename)
        and not os.path.exists(pwd + filename)
    ):
        fetch_file(full_path, filename)
        file_path = filename
        ARCHIVES[platform].append(filename)
    elif os.path.exists(filename):
        print("  -> Found: " + filename)
        file_path = filename
        ARCHIVES[platform].append(filename)
    elif os.path.exists(full_path):
        file_path = full_path
        print("  -> Found: " + file_path)
        ARCHIVES[platform].append(file_path)
    elif os.path.exists(os.path.join(parent, filename)):
        file_path = os.path.join(parent, filename)
        print("  -> Found: " + file_path)
        ARCHIVES[platform].append(file_path)
    elif os.path.exists(os.path.join(pwd, filename)):
        file_path = os.path.join(pwd, filename)
        print("  -> Found: " + file_path)
        ARCHIVES[platform].append(file_path)
    else:
        print("Parent: " + os.path.join(pwd, filename))
        print("  -> Artifact: " + filename)

def fetch_action(
    parent, manifest, component_filter, platform_filter, cuda_filter, retrieve
):
    for component in manifest.keys():
        if not "name" in manifest[component]:
            continue

        if component_filter is not None and component != component_filter:
            continue

        print("\n" + manifest[component]["name"] + ": " + manifest[component]["version"])

        for platform in manifest[component].keys():
            if "variant" in platform:
                continue

            if not platform in ARCHIVES:
                ARCHIVES[platform] = []

            if not isinstance(manifest[component][platform], str):
                if (
                    platform_filter is not None
                    and platform != platform_filter
                    and platform != "source"
                ):
                    print("  -> Skipping platform: " + platform)
                    continue

                if not "relative_path" in manifest[component][platform]:
                    for variant in manifest[component][platform].keys():
                        if cuda_filter is not None and variant != cuda_filter:
                            print("  -> Skipping variant: " + variant)
                            continue

                        parse_artifact(
                            parent,
                            manifest,
                            component,
                            platform,
                            retrieve,
                            variant,
                        )
                else:
                    parse_artifact(
                        parent, manifest, component, platform, retrieve
                    )

def post_action(output_dir, collapse=True):
    if len(ARCHIVES) == 0:
        return

    print("\nArchives:")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for platform in ARCHIVES:
        for archive in ARCHIVES[platform]:
            try:
                binTag = archive.split("-")[3].split("_")[1]
                # print(platform, binTag)
            except:
                binTag = None

            if re.search(r"\.tar\.", archive):
                print(":: tar: " + archive)
                tarball = tarfile.open(archive)
                topdir = os.path.commonprefix(tarball.getnames())
                tarball.extractall()
                tarball.close()

                print("  -> Extracted: " + topdir + "/")
                fix_permissions(topdir)

                if collapse:
                    flatdir = os.path.join(output_dir, platform)
                    flatten_tree(topdir, flatdir, binTag)
                    print("  -> Flattened: " + flatdir + "/")

            elif re.search(r"\.zip", archive):
                print(":: zip: " + archive)
                with zipfile.ZipFile(archive) as zippy:
                    topdir = os.path.commonprefix(zippy.namelist())
                    zippy.extractall()
                zippy.close()

                print("  -> Extracted: " + topdir)
                fix_permissions(topdir)

                if collapse:
                    flatdir = os.path.join(output_dir, platform)
                    flatten_tree(topdir, flatdir, binTag)
                    print("  -> Flattened: " + flatdir + "/")

    print("\nOutput: " + output_dir + "/")
    for item in sorted(os.listdir(output_dir)):
        if os.path.isdir(os.path.join(output_dir, item)):
            print(" - " + item + "/")
        elif os.path.isfile(os.path.join(output_dir, item)):
            print(" - " + item)

class DownloadWorker(QThread):
    finished = Signal(bool, str)
    
    def __init__(self, args):
        super().__init__()
        self.args = args
    
    def run(self):
        try:
            cmd = [
                sys.executable,
                sys.argv[0],
                "--download-only",
            ]
            
            for arg, value in vars(self.args).items():
                if value is not None:
                    cmd.extend([f"--{arg.replace('_', '-')}", str(value)])

            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )
            self.finished.emit(True, "")
        except subprocess.CalledProcessError as e:
            self.finished.emit(False, f"{str(e)}\nOutput: {e.output}")

class DownloaderGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NVIDIA Package Downloader")
        self.download_worker = None
        self.setup_ui()

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setSpacing(10)
        
        # Product selection
        product_layout = QHBoxLayout()
        product_label = QLabel("Product:")
        self.product_combo = QComboBox()
        self.product_combo.addItems(PRODUCTS.keys())
        self.product_combo.currentTextChanged.connect(self.on_product_change)
        product_layout.addWidget(product_label)
        product_layout.addWidget(self.product_combo)
        layout.addLayout(product_layout)

        # Release Label selection
        version_layout = QHBoxLayout()
        version_label = QLabel("Release Label:")
        self.version_combo = QComboBox()
        self.version_combo.addItems(CUDA_RELEASES.keys())
        version_layout.addWidget(version_label)
        version_layout.addWidget(self.version_combo)
        layout.addLayout(version_layout)

        # OS selection
        os_layout = QHBoxLayout()
        os_label = QLabel("Operating System:")
        self.os_combo = QComboBox()
        self.os_combo.addItems(OPERATING_SYSTEMS.keys())
        os_layout.addWidget(os_label)
        os_layout.addWidget(self.os_combo)
        layout.addLayout(os_layout)

        # Architecture selection
        arch_layout = QHBoxLayout()
        arch_label = QLabel("Architecture:")
        self.arch_combo = QComboBox()
        self.arch_combo.addItems(ARCHITECTURES.keys())
        arch_layout.addWidget(arch_label)
        arch_layout.addWidget(self.arch_combo)
        layout.addLayout(arch_layout)

        # Component selection
        comp_layout = QHBoxLayout()
        comp_label = QLabel("Component:")
        self.component_combo = QComboBox()
        self.component_combo.addItem("All Components")
        self.component_combo.addItems(COMPONENTS.keys())
        comp_layout.addWidget(comp_label)
        comp_layout.addWidget(self.component_combo)
        layout.addLayout(comp_layout)

        # Variant selection
        variant_layout = QHBoxLayout()
        variant_label = QLabel("CUDA Variant:")
        self.variant_combo = QComboBox()
        self.variant_combo.addItems(VARIANTS.keys())
        self.variant_combo.setEnabled(False)
        variant_layout.addWidget(variant_label)
        variant_layout.addWidget(self.variant_combo)
        layout.addLayout(variant_layout)

        # Output directory selection
        output_layout = QHBoxLayout()
        output_label = QLabel("Output Directory:")
        self.output_entry = QLineEdit()
        browse_button = QPushButton("Browse")
        browse_button.clicked.connect(self.browse_output)
        output_layout.addWidget(output_label)
        output_layout.addWidget(self.output_entry)
        output_layout.addWidget(browse_button)
        layout.addLayout(output_layout)

        # Command preview
        preview_label = QLabel("Command Preview:")
        self.command_text = QTextEdit()
        self.command_text.setReadOnly(True)
        self.command_text.setMaximumHeight(100)
        layout.addWidget(preview_label)
        layout.addWidget(self.command_text)

        # Download button
        self.download_button = QPushButton("Download")
        self.download_button.clicked.connect(self.execute_download)
        layout.addWidget(self.download_button)

        self.product_combo.currentTextChanged.connect(self.update_command_preview)
        self.version_combo.currentTextChanged.connect(self.update_command_preview)
        self.os_combo.currentTextChanged.connect(self.update_command_preview)
        self.arch_combo.currentTextChanged.connect(self.update_command_preview)
        self.component_combo.currentTextChanged.connect(self.update_command_preview)
        self.variant_combo.currentTextChanged.connect(self.update_command_preview)
        self.output_entry.textChanged.connect(self.update_command_preview)

        self.setMinimumWidth(600)
        self.setMinimumHeight(500)

    def on_product_change(self, product_text):
        is_cudnn = PRODUCTS[product_text] == "cudnn"
        
        self.variant_combo.setEnabled(is_cudnn)
        if not is_cudnn:
            self.variant_combo.setCurrentIndex(-1)
        
        self.component_combo.setEnabled(not is_cudnn)
        if is_cudnn:
            self.component_combo.setCurrentIndex(-1)
        
        self.version_combo.blockSignals(True)
        self.version_combo.clear()
        if is_cudnn:
            self.version_combo.addItems(CUDNN_RELEASES)
        else:
            self.version_combo.addItems(CUDA_RELEASES.keys())
        self.version_combo.blockSignals(False)
        
        self.update_command_preview()

    def browse_output(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if directory:
            self.output_entry.setText(directory)

    def update_command_preview(self):
        command = ["python", "download_cuda2.py"]
        
        product_text = self.product_combo.currentText()
        if product_text:
            product_key = PRODUCTS[product_text]
            command.extend(["--product", product_key])
        
        if self.version_combo.currentText():
            if PRODUCTS[self.product_combo.currentText()] == "cudnn":
                release_label = self.version_combo.currentText()
            else:
                release_label = CUDA_RELEASES.get(
                    self.version_combo.currentText(),
                    self.version_combo.currentText()
                )
            command.extend(["--label", release_label])
            
        if self.os_combo.currentText():
            os_key = OPERATING_SYSTEMS[self.os_combo.currentText()]
            command.extend(["--os", os_key])
        
        if self.arch_combo.currentText():
            arch_key = ARCHITECTURES[self.arch_combo.currentText()]
            command.extend(["--arch", arch_key])
        
        if (
            self.product_combo.currentText() != "cuDNN" and
            self.component_combo.currentText() != "All Components" and
            self.component_combo.currentText()
        ):
            component_key = COMPONENTS[self.component_combo.currentText()]
            command.extend(["--component", component_key])
        
        if self.variant_combo.isEnabled() and self.variant_combo.currentText():
            variant_key = VARIANTS[self.variant_combo.currentText()]
            command.extend(["--variant", variant_key])
        
        if self.output_entry.text():
            command.extend(["--output", self.output_entry.text()])
        
        self.command_text.setText(" ".join(command))


    def execute_download(self):
        command = self.command_text.toPlainText().strip()
        if command:
            self.download_button.setEnabled(False)
            
            args = argparse.Namespace()
            args.product = PRODUCTS[self.product_combo.currentText()]
            
            if PRODUCTS[self.product_combo.currentText()] == "cudnn":
                args.label = self.version_combo.currentText()
            else:
                args.label = CUDA_RELEASES.get(
                    self.version_combo.currentText(),
                    self.version_combo.currentText()
                )
            
            args.os = OPERATING_SYSTEMS[self.os_combo.currentText()]
            args.arch = ARCHITECTURES[self.arch_combo.currentText()]
            
            if self.variant_combo.isEnabled() and self.variant_combo.currentText():
                args.variant = VARIANTS[self.variant_combo.currentText()]
            else:
                args.variant = None
            
            if (
                self.product_combo.currentText() != "cuDNN" and
                self.component_combo.currentText() != "All Components" and
                self.component_combo.currentText()
            ):
                args.component = COMPONENTS[self.component_combo.currentText()]
            else:
                args.component = None
            
            args.output = self.output_entry.text() if self.output_entry.text() else "flat"
            
            self.download_worker = DownloadWorker(args)
            self.download_worker.finished.connect(self.on_download_complete)
            self.download_worker.start()
        else:
            QMessageBox.warning(
                self, 
                "Warning", 
                "Please configure the download options first."
            )


    def on_download_complete(self, success, error_message):
        self.download_button.setEnabled(True)
        if success:
            QMessageBox.information(self, "Success", "Download completed successfully!")
        else:
            QMessageBox.critical(self, "Error", f"Download failed: {error_message}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--product", help="Product name")
    parser.add_argument("--label", help="Release label version")
    parser.add_argument("--os", help="Operating System")
    parser.add_argument("--arch", help="Architecture")
    parser.add_argument("--component", help="Component name")
    parser.add_argument("--variant", help="Variant")
    parser.add_argument("--output", help="Output directory")
    
    args = parser.parse_args()

    if args.download_only:
        try:
            parent = f"{DOMAIN}/compute/{args.product}/redist/"
            manifest_uri = f"{parent}redistrib_{args.label}.json"
            
            manifest_response = urlopen(manifest_uri)
            manifest = json.loads(manifest_response.read())
            
            platform = f"{args.os}-{args.arch}"
            
            fetch_action(
                parent,
                manifest,
                args.component,
                platform,
                args.variant,
                True
            )
            
            post_action(args.output, True)
            
            sys.exit(0)
        except Exception as e:
            print(f"Error during download: {str(e)}", file=sys.stderr)
            sys.exit(1)
    else:
        app = QApplication(sys.argv)
        app.setStyle('Fusion')
        window = DownloaderGUI()
        window.show()
        sys.exit(app.exec())

if __name__ == "__main__":
    main()
