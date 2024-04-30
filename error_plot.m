clear
load('results.mat')

total = sum(error_results_cc,'all');
error_results_size = size(error_results_cc);

error_results_cc = double(error_results_cc)/1000;
error_results_ph_meas = double(error_results_ph_meas)/1000;
error_results_ph_rep = double(error_results_ph_rep)/1000;

step = 1/(error_results_size(1)+1);

%%
figure(1)
for k=1:6
    subplot(2,3,k)
    bar3(error_results_cc(:,:,k),1)
    set(gca,'XTickLabel',0:1:15)
    set(gca,'YTickLabel',0:0.25:1)
    if k < 4
        xlabel('# of Actual X Errors')
    else
        xlabel('# of Actual Z Errors')
    end
    if k < 4
        ylabel('X Error Rate')
    else
        ylabel('Z Error Rate')
    end
    zlabel('Relative Freq.')
    view(135,30)
end

subplot(2,3,1)
title('Freq. of Correctly Detected X Errors')
subplot(2,3,2)
title('Freq. of Incorrectly Detected X Errors w/o Logical Error')
subplot(2,3,3)
title('Freq. of Logical X Errors')
subplot(2,3,4)
title('Freq. of Correctly Detected Z Errors')
subplot(2,3,5)
title('Freq. of Incorrectly Detected Z Errors w/o Logical Error')
subplot(2,3,6)
title('Freq. of Logical Detected Z Errors')

sgtitle({'Code Capacity Model X and Z Errors for Various Qubit Error Rates','(n=1000)'})
saveas(gcf,'3D_plot_cc.png')

%%
figure(2)
for k=1:6
    subplot(2,3,k)
    bar3(error_results_ph_meas(:,:,k),1)
    set(gca,'XTickLabel',0:1:15)
    set(gca,'YTickLabel',0:0.25:1)
    if k < 4
        xlabel('# of Actual X Errors')
    else
        xlabel('# of Actual Z Errors')
    end
    ylabel('Measurement Error Rate')
    zlabel('Relative Freq.')
    view(135,30)
end

subplot(2,3,1)
title('Freq. of Correctly Detected X Errors')
subplot(2,3,2)
title('Freq. of Incorrectly Detected X Errors w/o Logical Error')
subplot(2,3,3)
title('Freq. of Logical X Errors')
subplot(2,3,4)
title('Freq. of Correctly Detected Z Errors')
subplot(2,3,5)
title('Freq. of Incorrectly Detected Z Errors w/o Logical Error')
subplot(2,3,6)
title('Freq. of Logical Detected Z Errors')

sgtitle({'Phenomenological Model X and Z Errors for Various Measurement Error Rates','(n=1000, Qubit Error Rate=0.1)'})
saveas(gcf,'3D_plot_ph_meas.png')

%%
figure(3)
for k=1:6
    subplot(2,3,k)
    bar3(error_results_ph_rep(:,:,k),1)
    set(gca,'XTickLabel',0:1:15)
    set(gca,'YTickLabel',1:2:31)
    if k < 4
        xlabel('# of Actual X Errors')
    else
        xlabel('# of Actual Z Errors')
    end
    ylabel('Measurement Repetitions')
    zlabel('Relative Freq.')
    view(135,30)
end

subplot(2,3,1)
title('Freq. of Correctly Detected X Errors')
subplot(2,3,2)
title('Freq. of Incorrectly Detected X Errors w/o Logical Error')
subplot(2,3,3)
title('Freq. of Logical X Errors')
subplot(2,3,4)
title('Freq. of Correctly Detected Z Errors')
subplot(2,3,5)
title('Freq. of Incorrectly Detected Z Errors w/o Logical Error')
subplot(2,3,6)
title('Freq. of Logical Detected Z Errors')

sgtitle({'Phenomenological Model X and Z Errors for Various Measurement Repetitions','(n=1000, Qubit Error Rate=0.1, Measurement Error Rate=0.1)'})
saveas(gcf,'3D_plot_ph_rep.png')