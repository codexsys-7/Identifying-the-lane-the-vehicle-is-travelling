![Lane](https://64.media.tumblr.com/8d0acd9d28dee73d59e981d420f7757c/b7a8b678c74aea84-3e/s540x810/f938c2d19e261e66adf4ab7207f2715b62eff733.gif)

# _**Identifying-the-lane-the-vehicle-is-travelling**_
Lane detection is a very vast area to work on, since a lot of efforts are being put on to avoid even minute error and making a safe self-driving environment sustainable and efficient. So, there are a lot of methods where we can implement this lane detection while driving, one method we have used is through OpenCV where using its in built functions such as Sobel operator, perspective transformation and edge detection we have done rather a basic lane detection model, which identifies the lane the vehicle is travelling.

# _**Base Paper**_
+ https://www.researchgate.net/publication/341460683_Lane_detection_technique_based_on_perspective_transformation_and_histogram_analysis_for_self-driving_cars
+ https://www.ijert.org/lane-line-detection-for-vehicles

# _**Project Approach**_
Specifically any algorithm was not used in this particular project, most of the lane detection was done using different types of filters such as Sobel, perspective tranformation, lane point detection. All these are integrated into one fucntion that is detect edges which will detect the edges by applying all the above filters. Then we have to fit the above method into a polynomial and also we have to calculate the curvature of the lane since some lane might be curved and some wont, so because of that we need to calcualte the curvature of the lane. Finally we have to process and render the image.

# _How to Execute?_
So, before execution we have some pre-requisites that we need to download or install i.e., anaconda environment, python and a code editor.
**Anaconda**: Anaconda is like a package of libraries and offers a great deal of information which allows a data engineer to create multiple environments and install required libraries easy and neat.

**Download link:**

![Anaconda](https://66.media.tumblr.com/5570e67a91bb118e3614194e23099079/tumblr_om107lgwrt1vbcnq8o1_500.gifv)

https://www.anaconda.com/

**Python**: Python is a most popular interpreter programming language, which is used in almost every field. Its syntax is very similar to English language and even children and learning it nowadays, due to its readability and easy syntax and large community of users to help you whenever you face any issues.

**Download link:**

![Python](https://i.pinimg.com/originals/24/97/b4/2497b48e8f9778fb8463c525e14794f9.gif)

https://www.python.org/downloads/

**Code editor**: Code editor is like a notepad for a programming language which allows user to write, run and execute program which we have written. Along with these some code editors also allows us to debug, which usually allows users to execute the code line by line and allows them to see where and how to solve the errors. But I personally feel visual code is very good to work with any programming language and makes a great deal of attachment with user.

**Download links:**

![Vs code](https://www.thisprogrammingthing.com/assets/headers/vscode@400.png) ![Pycharm](https://www.esoftner.com/wp-content/uploads/2019/12/PyCharm-Logo.png)

+ https://code.visualstudio.com/Download, 
+ https://www.jetbrains.com/pycharm/download/#section=windows

# How to create a new environment and configure jupyter notebook with it.
Let us define an environment and why we need different environments. An environment is a collection of libraries that are required to run our project. When we already have an environment with the necessary libraries, why do we need a new environment?
To avoid version mismatches, we create a new environment for each project. For example, in your previous project, you used "tf env" with tensorflow 2.4 and keras 2.4, but in your current project, you must use tensorflow 2.6 and keras 2.6. If you continue your project in the "tf env" environment, there will be a version mismatch and you will need to update tensorflow and keras, but this will cause problems with the previous project's execution. To avoid this, we create a new environment with tensorflow 2.6 and keras 2.6 and resume our project.

Let us now see how to create an environment in anaconda.
+ Type “conda create –n <<name_of_your_env>>”
example: conda create -n env
+ It will ask to proceed with the environment location, type ‘y’ and press enter.
+ When you press ‘y’, the environment will be created. To activate your environment type conda activate <<your_env_name>> . E.g., conda activate myenv.
+ You can see that the environment got changed after conda activate myenv line. It changed from “base” to “myenv” which means you are now working in “myenv” environment.
+ To install a library in your virtual environment type pip install <library_name>.
e.g., pip install pandas
+ Instead of installing libraries one by one you can even install by bunch, i.e., we have a txt file called requirements.tx which consists of all the libraries required to proceed with the project, so we can use it.
+ so, before installing requirements.txt, make sure you are in the specific path where your requirements.txt is located, basically this file is located in the folder where our executable files are located, so we need to move to that directory by following command.
**cd A:\folder_name**
+ Here A -> drive, folder name -> path where your executable file is saved
+ I go to that file path in anaconda using cd command 
1.	Go to drive where your project file is.
2.	Go to the path of your project using cd <path>
3.	Type pip install –r requirements.txt 
+ And all your required libraries will be downloaded and you can start your project.
+ But if you want to use jupyter notebook on the new environment you have to set it up for the new environment.
+ After you have installed all the libraries and created an environment, you need an editor to run the code, that is starting jupyter notebook, as soon as you enter jupyter notebook in the terminal you will definitely get this error. “Jupiter” is not recognized as an internal or external command.
So, to solve it it we have 2 commands.
1.	conda install –c conda-forge jupyterlab
2.	conda install –c anaconda python
Now you are ready to use jupyter on this environment and start with your project!

![thanks](https://media1.tenor.com/images/4d0f77f33c02ac812f0ba7827e3b7f44/tenor.gif?itemid=14352319)
  
### **Credits to my friend who gave detailed explanation of installation procedure.**
+ https://github.com/PaVaNTrIpAtHi
+ https://www.linkedin.com/in/pavan-tripathi-3993641a1/

# Steps to Run the code.
**Note:** Make sure you have added path while installing the software’s.
1. Install the prerequisites mentioned above.
2. open anaconda prompt and create a new environment.
  - conda create -n "env_name"
  - conda activate "env_name"
3. Install necessary libraries from requirements.txt file provided.
4. Run pip install -r requirements.txt or conda install requirements.txt (Requirements.txt is a text file consisting of all the necessary libraries required for executing this python file. If it gives any error while installing libraries, you might need to install them individually.)
5. Run main_code.ipynb run the final code, and make sure to change the path of the model and image folders.

# Data Description
This particular project dosent require any specific kind of dataset, we have used the existing filters to identify the lines in the lane and to get the curvature of the lane we have used polynomial fit method

 **Credits to the owners for giving the reference.**

# _Issues Faced._
1. We might face an issue while installing specific libraries.
2. Make sure you have the latest version of python, since sometimes it might cause version mismatch.
3. Adding path to environment variables in order to run python files and anaconda environment in code editor, specifically in visual studio code.
4. Refer to the Below link to get more details on installing python and anaconda and how to configure it.
+ https://techieyantechnologies.com/2022/06/get-started-with-creating-new-environment-in-anaconda-configuring-jupyter-notebook-and-installing-libraries-using-requirements-txt-2/
5. Make sure to change the paths of the images used in the code.
6. regarding Car, It was unable to correctly predict the lane curve at some point.
7. Try changing some parameters to best fit the lane curves.
8. You can use other Edge detection techniques instead of Sobel operator.
9. curve fitting code cell is little complicated to understand.

# _Note:_
**All the required data has been provided over here. Please feel free to contact me for any issues.**

### **Let’s Connect**
https://www.linkedin.com/in/abhinay-lingala-5a3ab7205/

![Connect](https://i.pinimg.com/originals/25/d7/27/25d7274b313939ee82ac5fd323298147.gif)

# _**Yes, you now have more knowledge than yesterday, Keep Going.**_
![Congrats](https://media1.tenor.com/images/c4998642cbd6eeb3e71d591d6e6abb8f/tenor.gif?itemid=15717977)
