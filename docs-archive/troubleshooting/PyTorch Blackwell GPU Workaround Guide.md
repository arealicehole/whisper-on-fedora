

# **Resolving PyTorch Incompatibility with the NVIDIA Blackwell Architecture on Fedora 42**

## **Executive Summary**

An investigation into errors encountered during a speaker diarization task has identified a root-cause incompatibility between a new NVIDIA GeForce RTX 5060 Ti graphics card and the PyTorch deep learning framework on a Fedora 42 operating system.1 The core of the issue is a predictable architectural mismatch. The RTX 5060 Ti is based on the next-generation Blackwell architecture, which features Compute Capability 12.0, a specification not supported by the pre-compiled kernels included in standard, stable releases of PyTorch.1 This situation is a common challenge for early adopters of next-generation hardware, representing a natural lag between the release of new hardware and the integration of its support into the stable software ecosystem.  
This technical report provides a comprehensive guide to resolving this incompatibility. It details three distinct and viable solution pathways, each tailored to the specific environment of Fedora 42\. These pathways are: (1) leveraging a pre-release PyTorch nightly build for immediate access to experimental hardware support; (2) compiling PyTorch directly from its source code for a definitive, system-optimized solution; and (3) deploying a containerized Docker environment for a highly stable, reproducible, and isolated workflow aligned with modern Machine Learning Operations (MLOps) practices.1  
For the specific goal of enabling the pyannote.audio speaker diarization task, the **PyTorch Nightly Build** is the most highly recommended solution. It offers the optimal balance of rapid implementation and immediate functionality, allowing for a swift return to the primary research or development objective.1 The report concludes with strategic best practices for ensuring long-term environmental stability, particularly the critical protocol of version-locking a working environment to prevent future breakages from software updates.

## **Deconstructing the Architectural Mismatch: Blackwell, Compute Capability 12.0, and PyTorch**

To effectively engineer a solution, it is essential to first establish a precise understanding of the technical factors causing the incompatibility. The error is not the result of a defect in either the hardware or the software but is a predictable consequence of the asynchronous development and release cycles of GPU architectures and the deep learning frameworks that rely on them.

### **Correcting the Record: From Ada Lovelace to Blackwell**

Initial analysis of the issue may have pointed to a misidentification of the GPU architecture. The NVIDIA GeForce RTX 5060 Ti is not a member of the Ada Lovelace family (found in the RTX 40 series) but is instead the vanguard of the newer **Blackwell** architecture.1 In the context of software compatibility, the marketing name of the architecture is less important than its technical specification, known as  
**Compute Capability (CC)**. This version number, assigned by NVIDIA, defines the hardware's core instruction set, available features, and the way compilers must target it for code generation.1  
The Ada Lovelace architecture is designated as Compute Capability 8.9. The Blackwell architecture introduces a significant generational leap to **Compute Capability 12.0**. This major version increment from 8 to 12 signifies a substantial evolution in hardware capabilities, including next-generation Tensor Cores and enhanced support for low-precision floating-point formats like FP8 and FP4, which are increasingly critical for accelerating AI workloads.1 The large jump in the major version number is a strong signal of a significant, and potentially breaking, architectural change, providing immediate context for the incompatibility challenges being faced. This pattern is part of a predictable technological cycle observed with each new GPU generation. The following table provides a clear reference for these architectural generations.  
**Table 1: NVIDIA Architecture and Compute Capability Reference**

| Architecture | GPU Series | Compute Capability (CC) | Key ML-Relevant Features |
| :---- | :---- | :---- | :---- |
| Turing | GeForce RTX 20 | 7.5 | First-gen RT Cores, Tensor Cores (INT8/INT4) |
| Ampere | GeForce RTX 30 | 8.0 / 8.6 | TF32, 2nd-gen RT Cores, Structured Sparsity |
| Ada Lovelace | GeForce RTX 40 | 8.9 | FP8, 3rd-gen RT Cores, DLSS 3 Frame Generation |
| Blackwell | GeForce RTX 50 | 12.0 | Next-gen Tensor Cores, Enhanced FP8/FP4 |

Data sourced from.1

### **Anatomy of a CUDA Error: "No Kernel Image is Available"**

When a user installs PyTorch using a standard package manager command like pip install torch, the downloaded binary is not a monolithic file. It contains a library of pre-compiled GPU programs, known as kernels, for a specific list of supported Compute Capabilities.1 This is a practical measure to keep the package size manageable while supporting the most common hardware in the market at the time of release.  
This compiled code exists in two primary forms:

* **SASS (Streaming Multiprocessor Assembly):** This is the final, native machine code for a specific GPU architecture (e.g., sm\_89 for Ada Lovelace). SASS offers the highest possible performance because it requires no further compilation at runtime. However, it is not forward-compatible; SASS compiled for one generation will not run on a future one. Stable PyTorch releases are built with SASS for all established architectures to guarantee maximum performance.1  
* **PTX (Parallel Thread Execution):** This is an intermediate, assembly-like language that is forward-compatible. When an application is run on a GPU for which no native SASS code is available, the NVIDIA driver can perform a just-in-time (JIT) compilation of the PTX code into SASS for the new hardware. While this provides flexibility, it introduces a noticeable startup delay and is often reserved for development or specific deployment scenarios.1

The error message CUDA error: no kernel image is available for execution on the device is the direct result of this system. The stable PyTorch 2.5.1 installation, developed and released before Blackwell GPUs were available, contains SASS kernels for architectures up to CC 9.0 (Hopper) but crucially lacks any code for sm\_120 (Blackwell). When PyTorch attempts to execute a CUDA operation, it searches its library for a matching kernel, finds none, and raises the fatal error.1 This reframes the user's experience from "my software is broken" to "I am operating in the gap between hardware release and stable software support," a crucial mindset for an advanced practitioner.

### **The Dependency Web: pyannote.audio and PyTorch Versioning**

The compatibility challenge extends to the application layer. While older versions of the pyannote.audio library were strictly pinned to outdated PyTorch versions (e.g., 1.10.x), which would be incompatible with any modern GPU, the current project is under active development.1 State-of-the-art pipelines, such as  
pyannote/speaker-diarization-3.1 hosted on Hugging Face, are designed to work with modern PyTorch versions. This means the resolution strategy is not to downgrade the application but to upgrade its underlying framework. The correct approach is to begin with a clean software environment, install the latest version of pyannote.audio, and then surgically replace the default PyTorch dependency it installs with a version that is explicitly compatible with the Blackwell architecture.1  
The user's anecdotal experience of the task "working before" is a critical clue. This prior success was likely the result of using a transient nightly software build that contained temporary or experimental support for the new hardware. The subsequent failure indicates that this experimental support was updated, changed, or refactored in a later nightly build—a common occurrence in a rapidly evolving development branch.1 This observation strongly validates that the nightly build channel is the correct place to find a solution and underscores the need to "lock" the version of a working build to prevent future updates from breaking the environment again.

## **Establishing a Resilient Foundation: System Configuration on Fedora 42**

Before any of the PyTorch-level solutions can be implemented, it is imperative to establish a stable and correctly configured foundation on the host operating system. Any errors or misconfigurations at this foundational driver and toolkit layer will prevent all subsequent efforts from succeeding. This section provides a meticulous guide to preparing the Fedora 42 environment, a process that is non-negotiable for a successful outcome.  
The challenge presented by this scenario can be understood as a "dependency sandwich." The user's application (pyannote.audio) is caught between two layers of incompatibility: an operating system that is too new for its required toolchain dependencies (Fedora 42's default GCC 15 is not supported by the CUDA Toolkit) and a hardware component that is too new for its application-level dependencies (the Blackwell GPU is not supported by stable PyTorch). The solution requires resolving the lower-level OS and driver issue first before the higher-level Python and PyTorch issue can be addressed.

### **NVIDIA Driver Management via RPM Fusion**

For Fedora-based systems, the community-standard and most robust method for installing the proprietary NVIDIA driver is through the RPM Fusion repository.1 This approach is strongly preferred over manual installation from  
.run files, which are brittle and tend to break with every system kernel update. RPM Fusion utilizes akmod (Automatic Kernel Module), a service that automatically rebuilds and signs the necessary NVIDIA kernel modules whenever the system kernel is updated, ensuring long-term stability and resilience.1  
The following procedure outlines the recommended installation process:

1. **Enable RPM Fusion Repositories:** Open a terminal and execute the following commands to add both the free and non-free RPM Fusion software sources to the system's package manager.  
   Bash  
   sudo dnf install https://download1.rpmfusion.org/free/fedora/rpmfusion-free-release-$(rpm \-E %fedora).noarch.rpm  
   sudo dnf install https://download1.rpmfusion.org/nonfree/fedora/rpmfusion-nonfree-release-$(rpm \-E %fedora).noarch.rpm

2. **Update the System:** To ensure all dependencies are aligned, perform a full system update to install the latest kernel and supporting packages, then reboot the machine.  
   Bash  
   sudo dnf update \-y  
   sudo reboot

3. **Install the NVIDIA Driver:** Install the akmod-nvidia package, which manages the kernel module, along with the associated CUDA driver libraries.  
   Bash  
   sudo dnf install akmod-nvidia xorg-x11-drv-nvidia-cuda

4. **Wait and Verify:** This is a critical step where patience is required. After the installation command completes, the akmod service begins building the NVIDIA kernel module in the background. This process can take several minutes. Rebooting prematurely is a common cause of failure. Wait for 5-10 minutes, and then verify that the module has been successfully built by running:  
   Bash  
   modinfo \-F version nvidia

   A successful build will return a driver version number (e.g., 580.76.05). If an error is returned, wait longer and check again before proceeding.1  
5. **Reboot and Final Verification:** Once the module is confirmed to be built, reboot the system one final time. After logging in, open a terminal and execute the nvidia-smi command. The successful display of a table containing details about the GeForce RTX 5060 Ti confirms that the driver is installed and communicating correctly with the hardware.1

### **CUDA Toolkit and Toolchain Management**

While the driver provides the necessary runtime API to execute CUDA applications, compiling software like PyTorch from source (Solution Pathway II) requires the full NVIDIA CUDA Toolkit, which includes the NVIDIA CUDA Compiler (nvcc).1 A significant challenge on a cutting-edge distribution like Fedora 42 is toolchain incompatibility. The default system compiler, GCC 15, is not yet supported by NVIDIA's CUDA Toolkit, which requires an older, compatible version.1  
The following steps detail how to install the CUDA Toolkit and implement the necessary compiler workaround:

1. **Add the CUDA Repository:** As an official CUDA repository for Fedora 42 may not be immediately available upon its release, the repository for the previous version, Fedora 41, can be used. This is a common and effective practice for early adopters.  
   Bash  
   sudo dnf config-manager \--add-repo https://developer.download.nvidia.com/compute/cuda/repos/fedora41/x86\_64/cuda-fedora41.repo

2. **Install a Compatible Compiler:** Install the GCC 14 toolchain, which is known to be compatible with the current CUDA Toolkit versions.  
   Bash  
   sudo dnf install gcc14 gcc14-c++

3. **Install the CUDA Toolkit:** With the repository and compatible compiler in place, install the latest available CUDA Toolkit.  
   Bash  
   sudo dnf install cuda-toolkit-12-9

4. **Configure Environment:** To make the CUDA compiler and tools accessible from the command line, their installation paths must be added to the shell's environment. Add the following lines to the end of the \~/.bashrc or \~/.zshrc file, then either restart the terminal or run source \~/.bashrc to apply the changes.  
   Bash  
   export PATH=/usr/local/cuda/bin:$PATH  
   export LD\_LIBRARY\_PATH=/usr/local/cuda/lib64:$LD\_LIBRARY\_PATH

With the driver and toolkit correctly installed and configured, the system's foundation is now stable and prepared for one of the application-level PyTorch solutions.

## **Solution Pathway I: The Agile Approach \- Leveraging PyTorch Nightly Builds**

For users who require the most direct path to enabling new hardware, the pre-release software channel offers the quickest resolution with the least amount of system configuration. This approach is highly recommended for the current objective of running the speaker diarization task.

### **Principles of Pre-Release Software**

PyTorch development is split into two primary distribution channels: "Stable" releases, which are thoroughly tested and intended for production use, and "Preview (Nightly)" builds, which are generated daily from the latest development codebase.1 Support for new hardware architectures like Blackwell appears in the nightly channel many months before it is considered stable enough for a general release. This channel is the front line where developers first add the necessary  
sm\_120 kernels. This represents a trade-off: gaining immediate access to cutting-edge features and hardware support at the cost of potential instability or undocumented changes inherent in pre-release software.1

### **Step-by-Step Implementation**

To avoid conflicts with system-level packages or other Python projects, it is a non-negotiable best practice to perform this installation within a dedicated, clean Python virtual environment.

1. **Environment Isolation:** Create and activate a new virtual environment using Python's built-in venv module.  
   Bash  
   python3 \-m venv pyannote-env  
   source pyannote-env/bin/activate

2. **Installation of the PyTorch Nightly Build:** Use pip to install the pre-release version of PyTorch. The command requires several specific flags: \--pre instructs pip to include pre-release versions in its search; \--index-url points to the official PyTorch nightly package repository; and the cu128 suffix (or a newer version like cu129) specifies the build compatible with the latest CUDA versions required for Blackwell support.1  
   Bash  
   pip3 install \--pre torch torchvision torchaudio \--index-url https://download.pytorch.org/whl/nightly/cu128

3. **Integration with pyannote.audio:** With the correct PyTorch nightly build installed, the final step is to install the application-level dependencies within the same activated environment.  
   Bash  
   pip install pyannote.audio huggingface\_hub transformers

   Depending on the specific diarization workflow, the openai-whisper package may also be required.  
4. **Pipeline Instantiation:** Modern pyannote.audio workflows involve downloading pre-trained models from the Hugging Face Hub, which requires authentication. The following Python snippet demonstrates how to load the state-of-the-art diarization pipeline and, critically, move it to the GPU for execution using pipeline.to(torch.device("cuda")).1  
   Python  
   from pyannote.audio import Pipeline  
   import torch

   \# A Hugging Face access token is required for this model  
   pipeline \= Pipeline.from\_pretrained(  
       "pyannote/speaker-diarization-3.1",  
       use\_auth\_token="YOUR\_HUGGINGFACE\_TOKEN\_HERE"  
   )

   pipeline.to(torch.device("cuda"))  
   print("Pipeline loaded successfully on GPU.")

### **The Verification Protocol**

To provide definitive confirmation that the entire hardware and software stack is functioning correctly, the following Python script should be executed within the activated virtual environment. This script serves as a comprehensive diagnostic tool to validate the solution.1

Python

import torch

try:  
    print(f"PyTorch Version: {torch.\_\_version\_\_}")  
      
    cuda\_available \= torch.cuda.is\_available()  
    print(f"CUDA Available: {cuda\_available}")

    if cuda\_available:  
        device\_count \= torch.cuda.device\_count()  
        print(f"Device Count: {device\_count}")  
          
        gpu\_name \= torch.cuda.get\_device\_name(0)  
        print(f"GPU: {gpu\_name}")  
          
        cc\_major, cc\_minor \= torch.cuda.get\_device\_capability(0)  
        print(f"Compute Capability: {cc\_major}.{cc\_minor}")  
          
        \# Final confirmation: perform a tensor operation on the GPU  
        tensor \= torch.rand(3, 3).to('cuda')  
        print("Tensor successfully created on GPU:")  
        print(tensor)  
        print("\\nEnvironment is configured correctly for Blackwell GPU.")  
    else:  
        print("\\nCUDA is not available. Please check driver and toolkit installation.")

except Exception as e:  
    print(f"\\nAn error occurred during verification: {e}")

A successful execution of this script will produce output confirming that CUDA is available, the PyTorch version string contains a .dev suffix indicating a nightly build, the GPU is correctly identified as the "NVIDIA GeForce RTX 5060 Ti," and—most importantly—the Compute Capability is reported as 12.0. The final, successful creation of a tensor on the 'cuda' device is the ultimate proof that the nightly build contains the necessary sm\_120 kernels and that the environment is ready for the diarization task.1

## **Solution Pathway II: The Definitive Approach \- Compiling PyTorch from Source**

For users who require maximum control, long-term stability independent of pre-release channels, or custom build configurations, compiling PyTorch directly from its source code is the most definitive solution. This process creates a version of the library that is specifically tailored to the host system's hardware and software environment, guaranteeing compatibility. This path should be chosen when nightly builds prove to be unstable or when specific optimizations are required.

### **Build Environment Preparation**

This advanced procedure assumes that the foundational system configuration detailed in Section III—including the NVIDIA driver, the full CUDA Toolkit, and the compatible GCC 14 toolchain—has been successfully completed. In addition, several build-time dependencies are required.

* **Install Build Prerequisites:** Use the dnf package manager to install the necessary development tools and libraries for compiling C++ and Python projects.  
  Bash  
  sudo dnf install git cmake ninja-build gcc-c++ make python3-devel

### **The Critical Directive: TORCH\_CUDA\_ARCH\_LIST**

The single most important step in this entire process is explicitly instructing the PyTorch build system which GPU architectures to compile native SASS kernels for. This is controlled by the TORCH\_CUDA\_ARCH\_LIST environment variable.1 While the build system may attempt to auto-detect the local GPU, setting this variable explicitly removes all ambiguity and ensures that the final binary contains the required  
sm\_120 machine code for the Blackwell architecture.

* **Set the Target Architecture:**  
  Bash  
  export TORCH\_CUDA\_ARCH\_LIST="12.0"

### **Complete Compilation Workflow**

The following sequence of shell commands provides a complete, annotated workflow for cloning the PyTorch repository, configuring the build environment with Fedora-specific settings, and executing the compilation process.

1. **Clone the PyTorch Repository:** It is essential to use the \--recursive flag to ensure that all required submodules are also cloned.  
   Bash  
   git clone \--recursive https://github.com/pytorch/pytorch  
   cd pytorch

2. **Configure the Build Environment:** This block of export commands sets up the shell environment for the build. It directs the build system to use the compatible GCC 14 compiler, explicitly targets the Blackwell architecture, enables CUDA support, and disables the building of tests to save a significant amount of time.1  
   Bash  
   \# Set compilers to the compatible GCC 14 for Fedora 42  
   export CC=/usr/bin/gcc-14  
   export CXX=/usr/bin/g++-14

   \# Explicitly target the Blackwell architecture (CC 12.0)  
   export TORCH\_CUDA\_ARCH\_LIST="12.0"

   \# Standard build flags to enable CUDA and disable unnecessary components  
   export USE\_CUDA=1  
   export BUILD\_TEST=0

3. **Build and Install PyTorch:** Execute the setup script to begin the compilation. This is a lengthy and resource-intensive operation that can take over an hour to complete, depending on the system's CPU and memory resources.1  
   Bash  
   python3 setup.py install

Upon completion, the setup.py install command will have built PyTorch from source and installed it directly into the currently active Python environment. To validate the custom build, execute the exact same Verification Protocol script from Section IV. The output should be identical, confirming that the self-compiled version recognizes the Blackwell GPU and can successfully execute CUDA operations.

## **Solution Pathway III: The Encapsulated Approach \- A Containerized MLOps Workflow**

A powerful and increasingly standard alternative that circumvents the complexities of the host system is to use a containerized environment. This approach, a best practice in modern Machine Learning Operations (MLOps), isolates the entire software stack—from operating system libraries and the CUDA Toolkit to Python and PyTorch—from the host operating system, leading to unparalleled stability and reproducibility.

### **The Philosophy of Containerization**

By using Docker in conjunction with the NVIDIA Container Toolkit, it is possible to run a pre-configured environment that is guaranteed to be compatible with NVIDIA GPUs. This method offers several profound advantages. First, it completely bypasses host system issues like the GCC 15 compiler incompatibility on Fedora 42, as the container includes its own compatible toolchain within a known-good OS base (typically a stable version of Ubuntu).1 Second, this solution is highly resilient to changes on the host system. A kernel update on Fedora, which would otherwise require the  
akmod driver module to be rebuilt, will not break the containerized PyTorch environment. This provides a level of stability that is difficult to achieve with a bare-metal installation, making it ideal for long-term projects and production-like workflows.1

### **Deployment with NVIDIA NGC**

NVIDIA provides professionally maintained, performance-tuned Docker containers for all major deep learning frameworks on its NGC (NVIDIA GPU Cloud) catalog. These containers are the recommended starting point for any containerized workflow, as they are tested and optimized for NVIDIA hardware.

1. **Install Docker and the NVIDIA Container Toolkit:** Follow the official documentation from Docker and NVIDIA to install the Docker engine and the NVIDIA Container Toolkit on the Fedora 42 host system. This is a standard, one-time setup procedure that enables Docker to securely access the system's GPUs.1  
2. **Pull the Latest PyTorch Container:** Browse the NGC Catalog online to find the tag for the latest available PyTorch container. Pull the image from the NGC container registry using the docker command. For example:  
   Bash  
   docker pull nvcr.io/nvidia/pytorch:25.06-py3

3. **Run the Container:** Launch an interactive session within the container. The command includes several important flags: \--gpus all is the directive from the NVIDIA Container Toolkit that grants the container access to the RTX 5060 Ti; \-it starts an interactive terminal session; \--rm ensures the container is automatically removed when exited; and the \-v flag mounts a local project directory into the container's /workspace directory, allowing for persistent storage of code, data, and models.1  
   Bash  
   docker run \--gpus all \-it \--rm \-v /path/to/your/project:/workspace nvcr.io/nvidia/pytorch:25.06-py3

### **Customization and Execution**

Once inside the container's interactive shell, the user is in a standard Linux environment that is already equipped with a compatible, GPU-enabled version of PyTorch and the CUDA Toolkit.

1. **Install Application Dependencies:** Within the container's shell, use pip to install pyannote.audio and its dependencies.  
   Bash  
   pip install pyannote.audio huggingface\_hub transformers

2. **Verify the Environment:** Navigate to the /workspace directory (which is linked to the local project folder) and run the same Verification Protocol script from Section IV. The script should execute successfully, confirming that the containerized PyTorch can access and utilize the Blackwell GPU. The speaker diarization task can now be run from within this stable, isolated, and reproducible environment.

## **Strategic Analysis and Final Recommendations**

Three distinct and viable pathways have been detailed to resolve the incompatibility between the NVIDIA RTX 5060 Ti and PyTorch on Fedora 42\. The choice between them depends on the user's specific priorities, technical comfort level, and project requirements. The solutions can be understood not just as fixes for a current problem, but as strategies that map to different phases of a machine learning project's lifecycle: the Nightly Build for rapid prototyping and research, Compiling from Source for deep optimization and customization, and the Docker Container for stable, reproducible deployment.

### **Comparative Analysis of Solutions**

The following table provides a summary and comparison of these solution pathways to aid in selecting the most appropriate method.  
**Table 2: Comparison of Proposed Solution Pathways**

| Solution | Pros | Cons | Best For... | Fedora 42 Complexity |
| :---- | :---- | :---- | :---- | :---- |
| PyTorch Nightly Build | Quickest to install; No compilation needed. | Can be unstable; may break with updates. | Rapid prototyping; users who want the latest features without compiling. | Low |
| Compile PyTorch from Source | Full control; stable until recompiled; optimized for the system. | Time-consuming; complex build process; requires managing toolchain. | Users needing a specific PyTorch commit or custom build flags. | High (due to GCC 14 requirement) |
| Docker Container (NGC) | Highly stable and reproducible; isolates dependencies from the host OS. | Requires Docker setup; larger disk footprint. | Production-like workflows; avoiding host system dependency issues. | Medium (one-time setup) |

Data sourced from.1

### **Primary Recommendation**

For the stated goal of running a speaker diarization task with pyannote.audio, the **PyTorch Nightly Build (Pathway I)** is the most highly recommended solution.1 It offers the best balance of ease of implementation and immediate functionality. This approach allows for a quick return to the primary objective—the diarization task itself—without the significant time overhead of a full source compilation or the initial setup complexity of a Docker environment. It is the most pragmatic and efficient choice for a practitioner focused on results.

### **Ensuring Long-Term Stability: The requirements.txt Protocol**

To mitigate the primary risk of the recommended approach—the volatility of nightly builds—a final best practice is essential for ensuring long-term stability. Once a working environment is established with a specific nightly build that successfully runs the diarization task, the exact state of that environment must be captured and version-controlled.1  
This is achieved with the pip freeze command, which records the precise version of every package installed in the virtual environment into a text file.

Bash

pip freeze \> requirements.txt

This command will create a requirements.txt file containing entries that pin the specific versions, including the exact nightly build of PyTorch (e.g., torch==2.8.0.dev20250715+cu128). By committing this file to version control alongside the project code, the environment becomes perfectly reproducible. In the future, this known-good environment can be recreated in a new virtual environment with a single command: pip install \-r requirements.txt. This protocol transforms a potentially volatile development setup into a stable, version-controlled asset, preventing accidental upgrades from breaking the workflow and ensuring consistent, reproducible results.1

#### **Works cited**

1. PyTorch Ada Lovelace GPU Workaround