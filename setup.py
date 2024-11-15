from setuptools import setup, find_packages

setup(
    name='rvc-py',
    version='0.1.0',
    description='AI Voice Conversion Project',
    author=' Ryouko-Yamanda65777',
    author_email='',
    url='https://github.com/Ryouko-Yamanda65777',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        # Main libraries
        'requests',
        'tensorboardX',
        'fairseq==0.12.2',
        'faiss-cpu==1.7.3',
        
        # Audio libraries
        'numpy==1.23.5',
        'ffmpeg-python>=0.2.0',
        'praat-parselmouth>=0.4.2',
        'pyworld==0.3.4',
        'torchcrepe==0.0.23',
        'pedalboard',
        'edge-tts',
        
        # Optimization
        'einops',
        'local-attention',
        
        # Interfaces and cloud services
        'gradio==5.3.0',
        'mega.py',
        'wget',
        'gdown',
    ],
    entry_points={
        'console_scripts': [
            'download_models=src..main:download_models'
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)
