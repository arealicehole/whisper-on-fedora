# **Resolving PyTorch and Blackwell GPU Incompatibility for Advanced AI Workloads on Fedora 42**

An incompatibility between a new NVIDIA GeForce RTX 5060 Ti graphics card and the PyTorch deep learning framework on a Fedora 42 operating system has been identified as the root cause of errors encountered during a speaker diarization task using pyannote.audio. This situation, while frustrating, is a common challenge for early adopters of next-generation hardware. The core of the issue lies in a fundamental architectural mismatch between the pre-compiled PyTorch binaries and the new GPU's instruction set.  
The user's description of the GPU as "adaove lace" with a designation like "ada120" points to a slight but critical misidentification. The RTX 5060 Ti is not based on the Ada Lovelace architecture (found in the RTX 40 series) but on the newer **Blackwell** architecture. The "120" corresponds to its **Compute Capability 12.0** (often denoted as sm\_120), a key technical specification that defines the hardware's features and capabilities. This distinction is the technical key to understanding and resolving the problem.  
This report provides a comprehensive, step-by-step guide to resolve this incompatibility. It will present three viable solution pathways, each tailored to the Fedora 42 environment and ranging from the most straightforward to the most robust. The phenomenon of the task "working before" was likely the result of using a transient nightly software build that contained experimental support, which has since been updated or changed, highlighting the volatile nature of software support for brand-new hardware.

## **The Architectural Mismatch: Understanding Blackwell's Compute Capability 12.0**

To effectively solve the compatibility error, it is essential to first understand the fundamental reason for its occurrence. The problem is not a defect in the hardware or software but a predictable gap between the release cycles of new GPU architectures and the deep learning frameworks that support them.

### **A Generational Leap: From Ada Lovelace to Blackwell**

NVIDIA categorizes its GPU architectures with a **Compute Capability (CC)** version number. This version dictates the hardware's core features, the set of instructions it supports, and how compilers should target it for optimal performance. The RTX 40 series GPUs, based on the Ada Lovelace architecture, have a Compute Capability of 8.9. The new RTX 5060 Ti is part of the Blackwell family, which introduces the next major architectural version: **Compute Capability 12.0**. This significant jump in the major version number signifies a new generation of hardware features, including next-generation Tensor Cores and enhanced support for floating-point formats like FP8 and FP4, which are critical for AI workloads.  
The following table clarifies the architectural generations, correcting the initial confusion and establishing Compute Capability as the key metric for software compatibility.  
**Table 1: NVIDIA Architecture and Compute Capability Reference**

| Architecture | GPU Series | Compute Capability (CC) | Key ML-Relevant Features |
| :---- | :---- | :---- | :---- |
| Turing | GeForce RTX 20 | 7.5 | First-gen RT Cores, Tensor Cores (INT8/INT4) |
| Ampere | GeForce RTX 30 | 8.0 / 8.6 | TF32, 2nd-gen RT Cores, Structured Sparsity |
| Ada Lovelace | GeForce RTX 40 | 8.9 | FP8, 3rd-gen RT Cores, DLSS 3 Frame Generation |
| **Blackwell** | **GeForce RTX 50** | **12.0** | **Next-gen Tensor Cores, Enhanced FP8/FP4** |

### **Deconstructing the Error: "No Kernel Image is Available"**

When PyTorch is installed from a pre-compiled binary (e.g., via pip install torch), the package includes compiled GPU code, known as kernels, for a specific list of Compute Capabilities. This is done to keep the package size manageable while supporting the most common hardware. This compiled code exists in two primary forms: PTX, an intermediate, forward-compatible assembly-like language, and SASS, the final, GPU-specific machine code. For maximum performance, stable PyTorch releases are built with SASS for established architectures (e.g., sm\_75 for Turing, sm\_86 for Ampere, sm\_90 for Hopper).  
The error message CUDA error: no kernel image is available for execution on the device means that when PyTorch attempted to execute a command on the RTX 5060 Ti, it looked for a pre-compiled SASS kernel for sm\_120, found none in the installed package, and consequently failed. PyTorch version 2.5.1, and even much newer stable releases, were developed and released before Blackwell GPUs were widely available. Therefore, the developers did not include sm\_120 kernels in those builds, which typically support architectures only up to sm\_90 (Hopper).

### **The Dependency Web: pyannote.audio and PyTorch Versioning**

The compatibility challenge extends beyond just PyTorch and the GPU; the application layer, pyannote.audio, also has its own versioning requirements. Older documentation and requirement files for pyannote.audio show that it was strictly pinned to much older PyTorch versions, such as 1.10.x or 1.11.0. Attempting to use these outdated library versions will fail on modern hardware, as they predate even the Ada Lovelace architecture, let alone Blackwell.  
However, the current pyannote.audio project, particularly the state-of-the-art pipelines like pyannote/speaker-diarization-3.1 hosted on Hugging Face, is under active development. This implies compatibility with modern PyTorch versions. Therefore, the first step in any successful resolution must be to ensure the project's dependencies are not locked to an old, incompatible version of PyTorch. The correct approach is to begin with a clean software environment, install the latest pyannote.audio, and then replace the version of PyTorch it installs by default with one that supports the Blackwell architecture, as detailed in the following sections.

## **Foundational Integrity: Configuring the Fedora 42 Environment**

Before addressing the PyTorch incompatibility, it is critical to establish a stable and correctly configured foundation on the Fedora 42 operating system. This involves installing the proprietary NVIDIA driver and the full CUDA Toolkit. Errors at this foundational stage will prevent any of the subsequent PyTorch-level solutions from functioning correctly.

### **Best Practices for NVIDIA Driver Management via RPM Fusion**

For Fedora systems, the community-standard and most reliable method for installing proprietary NVIDIA drivers is through the RPM Fusion repository. This method uses akmod (Automatic Kernel Module) to automatically build and sign the necessary kernel modules, ensuring they remain compatible even after system kernel updates. This is a more robust and maintainable approach than manual installation from .run files, which can break with every kernel update.  
The following steps outline the recommended procedure:

1. **Enable RPM Fusion Repositories:** Open a terminal and execute the following commands to add both the free and non-free RPM Fusion repositories to the system's package manager.  
   `sudo dnf install https://download1.rpmfusion.org/free/fedora/rpmfusion-free-release-$(rpm -E %fedora).noarch.rpm`  
   `sudo dnf install https://download1.rpmfusion.org/nonfree/fedora/rpmfusion-nonfree-release-$(rpm -E %fedora).noarch.rpm`

2. **Update the System:** Ensure the system is running the latest kernel before installing the driver modules. Run a full system update and then reboot.  
   `sudo dnf update -y`  
   `sudo reboot`

3. **Install the Driver:** Install the akmod-nvidia package, which will manage the kernel module, and the associated CUDA driver libraries.  
   `sudo dnf install akmod-nvidia xorg-x11-drv-nvidia-cuda`

4. **Wait and Verify:** This is a critical step. After the installation command completes, the akmod service builds the kernel module in the background. This process can take several minutes. Rebooting prematurely is a common cause of failure. Wait 5-10 minutes, then verify that the module has been built successfully by running the following command:  
   `modinfo -F version nvidia`  
   If this command returns a driver version number (e.g., 580.76.05), the build is complete and it is safe to proceed. If it returns an error, wait longer and check again.  
5. **Reboot and Final Verification:** Once the module is built, reboot the system. After logging back in, open a terminal and run nvidia-smi. This command should display a table with details about the RTX 5060 Ti and the installed driver version, confirming a successful installation.

### **Installing the CUDA Toolkit and a Compatible Toolchain**

While the driver provides the runtime API necessary to run CUDA applications, compiling software like PyTorch from source requires the full CUDA Toolkit, which includes the NVIDIA CUDA Compiler (nvcc). A significant challenge on a cutting-edge distribution like Fedora 42 is toolchain incompatibility; the default GCC 15 compiler is not yet supported by NVIDIA's CUDA Toolkit, which requires an older version.  
The following steps detail how to install the CUDA Toolkit and work around this incompatibility:

1. **Add the CUDA Repository:** Since an official CUDA repository for Fedora 42 may not be immediately available, the repository for Fedora 41 can be used. This is a common practice for early adopters of new Fedora releases.  
   `sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/fedora41/x86_64/cuda-fedora41.repo`

2. **Install a Compatible Compiler:** Install the GCC 14 toolchain, which is compatible with the current CUDA Toolkit.  
   `sudo dnf install gcc14 gcc14-c++`

3. **Install the CUDA Toolkit:** Install the latest available CUDA Toolkit from the newly added repository.  
   `sudo dnf install cuda-toolkit-12-9`

4. **Configure Environment:** To make the CUDA tools accessible from the command line, their paths must be added to the shell's environment. Add the following lines to the end of the \~/.bashrc or \~/.zshrc file, then restart the terminal or run source \~/.bashrc.  
   `export PATH=/usr/local/cuda/bin:$PATH`  
   `export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH`

With the driver and toolkit correctly installed, the system is now prepared for one of the PyTorch-level solutions.

## **The Primary Solution: Harnessing PyTorch Nightly Builds**

For users who require immediate access to support for new hardware, the most direct path is often through pre-release software channels. This approach provides the quickest resolution with the least amount of system configuration.

### **Navigating the Bleeding Edge: Stable vs. Nightly Builds**

PyTorch maintains two primary distribution channels: "Stable" for production-ready, thoroughly tested versions, and "Preview (Nightly)" for builds generated daily from the latest development code. Support for new hardware architectures like Blackwell appears in the nightly channel months before it is incorporated into a stable release. This directly explains why the diarization task may have worked previously; it was likely running on a specific nightly build that had temporary or early support for the hardware.

### **Installation and Environment Setup**

To avoid conflicts with system packages or other projects, it is strongly recommended to perform this installation within a dedicated, clean Python virtual environment.

1. **Create a Clean Environment:** Use Python's built-in venv module to create and activate a new environment.  
   `python3 -m venv pyannote-env`  
   `source pyannote-env/bin/activate`

2. **Install the PyTorch Nightly:** Use pip to install the pre-release version of PyTorch. The command must include the \--pre flag to allow pip to find nightly versions and specify the index URL for the desired CUDA version. Based on recent forum discussions, the latest CUDA builds (e.g., cu128 or cu129) are required for Blackwell support.  
   `pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128`

### **Re-integrating the Diarization Workflow**

With the correct PyTorch nightly build installed, the final step is to install the application-level dependencies within the same activated environment.

1. **Install pyannote.audio:** Install the latest versions of pyannote.audio and its key dependencies from PyPI.  
   `pip install pyannote.audio huggingface_hub transformers`  
   Note: The whisper model is often used in conjunction with pyannote, so installing openai-whisper may also be necessary depending on the exact workflow.  
2. **Instantiate the Pipeline:** The modern usage pattern for pyannote.audio involves authenticating with Hugging Face to download pre-trained models. A small Python script can be used to load the pipeline.  
   `from pyannote.audio import Pipeline`  
   `import torch`

   `# Ensure you have a Hugging Face access token`  
   `pipeline = Pipeline.from_pretrained(`  
       `"pyannote/speaker-diarization-3.1",`  
       `use_auth_token="YOUR_HUGGINGFACE_TOKEN_HERE"`  
   `)`  
   `pipeline.to(torch.device("cuda"))`  
   `print("Pipeline loaded successfully on GPU.")`

### **Verification Protocol**

To provide a definitive confirmation that the entire software and hardware stack is functioning correctly, the following Python script should be executed within the activated virtual environment.  
`import torch`

`try:`  
    `print(f"PyTorch Version: {torch.__version__}")`  
      
    `cuda_available = torch.cuda.is_available()`  
    `print(f"CUDA Available: {cuda_available}")`

    `if cuda_available:`  
        `device_count = torch.cuda.device_count()`  
        `print(f"Device Count: {device_count}")`  
          
        `gpu_name = torch.cuda.get_device_name(0)`  
        `print(f"GPU: {gpu_name}")`  
          
        `cc = torch.cuda.get_device_capability(0)`  
        `print(f"Compute Capability: {cc}.{cc}")`  
          
        `# Final confirmation: perform a tensor operation on the GPU`  
        `tensor = torch.rand(3, 3).to('cuda')`  
        `print("Tensor successfully created on GPU:")`  
        `print(tensor)`  
        `print("\nEnvironment is configured correctly for Blackwell GPU.")`  
    `else:`  
        `print("\nCUDA is not available. Please check driver and toolkit installation.")`

`except Exception as e:`  
    `print(f"\nAn error occurred during verification: {e}")`

A successful execution of this script will report that CUDA is available, identify the GPU as the "NVIDIA GeForce RTX 5060 Ti", show a Compute Capability of (12, 0), and print a randomly generated tensor. This confirms that the nightly build includes the necessary sm\_120 kernels and the environment is ready for the diarization task.

## **The Definitive Solution: Compiling PyTorch from Source**

For users who require maximum control, long-term stability, or find the nightly builds to be unreliable, compiling PyTorch directly from its source code is the most definitive solution. This process builds a version of the library specifically tailored to the host system and its hardware, guaranteeing compatibility.

### **Preparing the Build Environment on Fedora**

In addition to the NVIDIA driver and CUDA Toolkit configured in Section 2, several build-time dependencies are required.

1. **Install Build Prerequisites:** Use dnf to install the necessary development tools and libraries.  
   `sudo dnf install git cmake ninja-build gcc-c++ make python3-devel`

2. **Confirm Foundational Tools:** Verify that the CUDA Toolkit is installed in /usr/local/cuda and that the GCC 14 compiler suite is available, as these are non-negotiable prerequisites for a successful build on Fedora 42\.

### **The Critical Build Flag: TORCH\_CUDA\_ARCH\_LIST**

The most important step in this process is explicitly telling the PyTorch build system which GPU architectures to compile native SASS kernels for. This is controlled by the TORCH\_CUDA\_ARCH\_LIST environment variable. While the build system may attempt to auto-detect the local GPU, setting this variable explicitly removes any ambiguity and ensures the final binary contains the required sm\_120 code.  
To target the Blackwell architecture of the RTX 5060 Ti, this variable must be set as follows : export TORCH\_CUDA\_ARCH\_LIST="12.0"

### **Step-by-Step Compilation Guide**

The following sequence of shell commands provides a complete workflow for cloning, configuring, and building PyTorch from source on Fedora 42\.

1. **Clone the PyTorch Repository:** Clone the official PyTorch repository and its submodules. The \--recursive flag is essential.  
   `git clone --recursive https://github.com/pytorch/pytorch`  
   `cd pytorch`

2. **Configure the Build Environment:** Set all necessary environment variables. This block incorporates the Fedora-specific workaround for the GCC toolchain and explicitly targets the Blackwell architecture.  
   `# Set compilers to the compatible GCC 14`  
   `export CC=/usr/bin/gcc-14`  
   `export CXX=/usr/bin/g++-14`

   `# Explicitly target the Blackwell architecture (CC 12.0)`  
   `export TORCH_CUDA_ARCH_LIST="12.0"`

   `# Standard build flags to enable CUDA and disable unnecessary components`  
   `export USE_CUDA=1`  
   `export BUILD_TEST=0` 

3. **Build and Install PyTorch:** Execute the setup script to begin the compilation and installation process. This is a lengthy operation that can take over an hour, depending on the system's CPU and memory.  
   `python3 setup.py install`

### **Installation and Validation of the Custom Build**

The setup.py install command will build PyTorch and install it directly into the currently active Python environment. To validate the custom build, execute the exact same Verification Protocol script from Section 3.4. The output should be identical, confirming that the self-compiled version recognizes the Blackwell GPU and can execute CUDA operations successfully.

## **The Encapsulated Solution: A Containerized Docker Workflow**

A powerful alternative that circumvents host system complexities is to use a containerized environment. This approach, a best practice in modern Machine Learning Operations (MLOps), isolates the entire software stack—from OS libraries and the CUDA Toolkit to Python and PyTorch—from the host operating system.

### **Isolating Complexity with Containers**

By using Docker in conjunction with the NVIDIA Container Toolkit, it is possible to run a pre-configured environment that is guaranteed to be compatible with NVIDIA GPUs. This method bypasses issues like the host system's GCC version, as the container includes its own compatible toolchain. Furthermore, this solution is highly resilient to host system changes. A kernel update on Fedora, which might otherwise require the akmod driver module to be rebuilt, will not break the containerized PyTorch environment. This provides a level of stability and reproducibility that is difficult to achieve on a bare-metal installation.

### **Deploying the Official PyTorch NGC Container**

NVIDIA provides professionally maintained, performance-tuned Docker containers for all major deep learning frameworks on its NGC (NVIDIA GPU Cloud) catalog. These containers are the recommended starting point for any containerized workflow.

1. **Install Docker and the NVIDIA Container Toolkit:** Follow the official documentation to install the Docker engine and the NVIDIA Container Toolkit on Fedora. This is a standard procedure that enables Docker to access the system's GPUs.  
2. **Pull the Latest PyTorch Container:** Browse the NGC Catalog to find the latest available PyTorch container tag. Pull the image using the docker command. For example :  
   `docker pull nvcr.io/nvidia/pytorch:25.06-py3`

3. **Run the Container:** Launch an interactive session within the container. The \--gpus all flag grants the container access to the RTX 5060 Ti, and the \-v flag mounts a local project directory into the container's /workspace directory, allowing for persistent storage of code and data.  
   `docker run --gpus all -it --rm -v /path/to/your/project:/workspace nvcr.io/nvidia/pytorch:25.06-py3`

### **Customizing the Container for Diarization**

Once inside the container's interactive shell, it functions as a standard Linux environment that is already equipped with a compatible version of PyTorch and the CUDA Toolkit.

1. **Install Application Dependencies:** Within the container's shell, use pip to install pyannote.audio and its dependencies.  
   `pip install pyannote.audio huggingface_hub transformers`

2. **Verify the Environment:** Navigate to the /workspace directory (which is linked to the local project folder) and run the Verification Protocol script from Section 3.4. The script should execute successfully, confirming that the containerized PyTorch can access and utilize the Blackwell GPU. The speaker diarization task can now be run from within this stable, isolated environment.

## **Conclusion: Strategic Recommendations for Long-Term Stability**

Three distinct and viable pathways have been detailed to resolve the incompatibility between the NVIDIA RTX 5060 Ti (Blackwell architecture) and PyTorch on Fedora 42\. These solutions—using a nightly build, compiling from source, or deploying a Docker container—cater to different technical requirements and user preferences.  
The following table provides a summary and comparison of these solution pathways to aid in selecting the most appropriate method.  
**Table 2: Comparison of Proposed Solution Pathways**

| Solution | Pros | Cons | Best For... | Fedora 42 Complexity |
| :---- | :---- | :---- | :---- | :---- |
| **PyTorch Nightly Build** | Quickest to install; No compilation needed. | Can be unstable; may break with updates. | Rapid prototyping; users who want the latest features without compiling. | Low |
| **Compile PyTorch from Source** | Full control; stable until recompiled; optimized for the system. | Time-consuming; complex build process; requires managing toolchain. | Users needing a specific PyTorch commit or custom build flags. | High (due to GCC 14 requirement) |
| **Docker Container (NGC)** | Highly stable and reproducible; isolates dependencies from the host OS. | Requires Docker setup; larger disk footprint. | Production-like workflows; avoiding host system dependency issues. | Medium (one-time setup) |

For the stated goal of running a speaker diarization task with pyannote.audio, the **PyTorch Nightly Build (Section 3\)** is the most highly recommended solution. It offers the best balance of ease of implementation and immediate functionality, allowing for a quick return to the primary objective without the overhead of a full source compilation or Docker setup.  
To ensure long-term stability and prevent future issues similar to the one that prompted this inquiry, a final best practice is advised. Once a working environment is established with a specific nightly build, the exact package versions should be captured in a requirements.txt file using the command pip freeze \> requirements.txt. This file will pin the precise version of torch (e.g., torch==2.8.0.dev...). By using this file to recreate the environment in the future (pip install \-r requirements.txt), accidental upgrades that could break the setup are prevented, providing a stable and reproducible development environment.

#### **Works cited**

1\. nvidia gpu compute capability reference \- mike bommarito, https://michaelbommarito.com/wiki/programming/tools/gpu-compute-capability/ 2\. RTX 5090 not working with PyTorch and Stable Diffusion (sm\_120 unsupported), https://forums.developer.nvidia.com/t/rtx-5090-not-working-with-pytorch-and-stable-diffusion-sm-120-unsupported/338015 3\. PyTorch support for sm\_120: NVIDIA GeForce RTX 5060, https://discuss.pytorch.org/t/pytorch-support-for-sm-120-nvidia-geforce-rtx-5060/220941 4\. CUDA GPU Compute Capability | NVIDIA Developer, https://developer.nvidia.com/cuda-gpus 5\. Ada Lovelace (microarchitecture) \- Wikipedia, https://en.wikipedia.org/wiki/Ada\_Lovelace\_(microarchitecture) 6\. 1\. NVIDIA Ada GPU Architecture Tuning Guide \- NVIDIA Docs Hub, https://docs.nvidia.com/cuda/ada-tuning-guide/index.html 7\. Rtx 5090 \- GPU \- Hardware \- NVIDIA Developer Forums, https://forums.developer.nvidia.com/t/rtx-5090/331369 8\. NVIDIA Ada GPU Architecture Compatibility Guide for CUDA Applications, https://docs.nvidia.com/cuda/ada-compatibility-guide/ 9\. raw \- Hugging Face, https://huggingface.co/KIFF/pyannote-speaker-diarization-endpoint/raw/2ea1bd172ce6d7fd22edbc53dcab101dc235184a/requirements.txt 10\. pyannote.audio \- PyPI, https://pypi.org/project/pyannote.audio/2.0.1/ 11\. pyannote/pyannote-audio: Neural building blocks for speaker diarization: speech activity detection, speaker change detection, overlapped speech detection, speaker embedding \- GitHub, https://github.com/pyannote/pyannote-audio 12\. Fedora Workstation 42 \- Nvidia Drivers \- YouTube, https://www.youtube.com/watch?v=2YeebhfRSx4 13\. Howto/NVIDIA \- RPM Fusion, https://rpmfusion.org/Howto/NVIDIA 14\. Fedora 42/41/40 NVIDIA Drivers Install Guide \[580.76.05 / 570.181 / 550.163.01 / 470.256.02 / 390.157 / 340.108\] \- If Not True Then False, https://www.if-not-true-then-false.com/2015/fedora-nvidia-guide/ 15\. How to Install Nvidia Drivers on Fedora Linux \- Tecmint, https://www.tecmint.com/install-nvidia-drivers-in-linux/ 16\. fedora 42 kde nvidia drivers \- Reddit, https://www.reddit.com/r/Fedora/comments/1k39z3j/fedora\_42\_kde\_nvidia\_drivers/ 17\. CUDA 12.9 on Fedora 42 Guide including getting \`cuda-samples ..., https://forum.level1techs.com/t/cuda-12-9-on-fedora-42-guide-including-getting-cuda-samples-running/230769 18\. Fedora 42 \-- CUDA toolkit from Nvidia, http://kofa.mmto.arizona.edu/fedora/f42\_cuda.html 19\. How do I install CUDA in fedora \- Reddit, https://www.reddit.com/r/Fedora/comments/18511p3/how\_do\_i\_install\_cuda\_in\_fedora/ 20\. Get Started \- PyTorch, https://pytorch.org/get-started/locally/ 21\. How Do I use Pytorch with RTX 5060 Ti, https://discuss.pytorch.org/t/how-do-i-use-pytorch-with-rtx-5060-ti/220926 22\. Building PyTorch with LibTorch From Source with CUDA Support \- Data Science \<3 Machine Learning, https://michhar.github.io/how-i-built-pytorch-gpu/ 23\. Pytorch Installation for different CUDA architectures \- Stack Overflow, https://stackoverflow.com/questions/68496906/pytorch-installation-for-different-cuda-architectures 24\. PyTorch for Cuda 12, https://discuss.pytorch.org/t/pytorch-for-cuda-12/169447 25\. Building from source: nvcc fatal : Unsupported gpu architecture 'compute\_120' \- C++, https://discuss.pytorch.org/t/building-from-source-nvcc-fatal-unsupported-gpu-architecture-compute-120/222262 26\. torch \- PyPI, https://pypi.org/project/torch/ 27\. Running PyTorch \- NVIDIA Docs \- NVIDIA Docs Hub, https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/running.html 28\. Need guidance on pytorch dependency size reduction inside docker image using nvidia-container-toolkit, https://discuss.pytorch.org/t/need-guidance-on-pytorch-dependency-size-reduction-inside-docker-image-using-nvidia-container-toolkit/218091 29\. PyTorch | NVIDIA NGC, https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch