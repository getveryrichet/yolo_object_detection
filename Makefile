build_image:
	docker build -t object-detection .

run.simple:
	docker run -v $(CURDIR):/home/richet -w /home/richet object-detection python detect_simple.py /home/richet/$(FILENAME) $(OUTPUTFILENAME)

download:
	sudo rm -r checkpoints || true
	sudo rm -r ._checkpoints || true
	sudo rm -r __MACOSX || true
	rm checkpoints.zip || true
	sudo rm -r data || true
	rm data.zip || true
	docker run -v $(CURDIR):/home/richet -w /home/richet object-detection gdown https://drive.google.com/uc\?id\=1FgbWJ8EZyMPJ1_JiHEFfobL-dNi3IAtT
	unzip checkpoints.zip
	docker run -v $(CURDIR):/home/richet -w /home/richet object-detection gdown https://drive.google.com/uc\?id\=1Un2S6rr7K-iKBNtOhOTSe8uKOQvs48wt
	unzip data.zip

	rm checkpoints.zip || true
	rm data.zip || true
conda.build.env:
	conda env remove --name yolo_python || true 
	conda env create --file environment.yml
	conda activate yolo_python 