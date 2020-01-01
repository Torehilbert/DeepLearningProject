python "C:\Source\DeepLearningProject\ASSETS\plot_timing.py" --data_path="D:\DTU\02456 Deep Learning\Project\Data\Experiment Discount Factor"^
 --x=Episodes --groups=[8:16,16:24,24:32,32:40,40:48] --xrange=[1,40000] --xres=100 --yrange=[0,280] --xlabel="Episodes"^
 --ylabel="Validation Reward" --title="LunarLander-v2" --legends=[$\gamma=0.96$_$\gamma=0.97$_$\gamma=0.98$_$\gamma=0.99$_$\gamma=1.00$] --ltypes=[:_:_-._-_:]
pause