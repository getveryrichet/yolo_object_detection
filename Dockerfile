FROM richet/ubuntu-richet:latest

# Configuring access to Jupyter
RUN echo "c.NotebookApp.password = u'sha1:f193d081b73e:b24457f756bb19fb90771883777b3705448c27cc'" >> /root/.jupyter/jupyter_notebook_config.py

ADD . /home/richet/
WORKDIR /home/richet

RUN apt-get -qq -y update && apt-get install libgl1-mesa-glx -y
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# # Jupyter listens port: 8888
# EXPOSE 8888
# Run Jupytewr notebook as Docker main process
CMD ["/virtualenv/trading_envs/bin/jupyter-notebook", "--allow-root", "--notebook-dir=/home/richet/notebooks", "--ip='0.0.0.0'", "--port=8888", "--no-browser"]