import numpy as np
import matplotlib.pyplot as plots

class Plotter():
    """
    The Plotter class is used to plot the results from the experiment reports.
    """
    def __init__(self, experiment_dir):
        self.cwd = experiment_dir

###############################################################################
# open funtions
###############################################################################

    def open_report(self):
        """
        Opens the csv file.
        """
        train_reports_dir = self.cwd + '/training_reports'
        test_reports_dir = self.cwd + '/test_reports'
        with open( self.cwd, mode=r) as f:
            reader = csv.reader(f)

        multireports = []
        feature_vector = []
        for parameter in sweepReport:
            multiReport = sweepReport[parameter]
            multireports.append(multiReport)
            feature_vector.append(parameter)
        return multireports, feature_vector


###############################################################################
# save method
###############################################################################

    def _save_plot(self, save_path, name):
        now = datetime.datetime.now()
        file_string_png = '{}/plots/png/{}_{}.png'.format(save_path, now.strftime('%Y%m%d_%H%M%S'), name)
        file_string_pdf = '{}/plots/pdf/{}_{}.pdf'.format(save_path, now.strftime('%Y%m%d_%H%M%S'), name)
        print('saving png format')
        print(file_string_png)
        plt.savefig(file_string_png)
        print('saving pdf format')
        plt.savefig(file_string_pdf)
        print(file_string_pdf)

###############################################################################
# statistics calculation
###############################################################################

    def training_statistics(self, multiReport):
        """
        Gets a multiReport and calculates mean and standard deviation of the test_report.
        """
        training_reports, _, _ = self.open_multiReport(multiReport)
        mean_vector = [np.mean(i) for i in zip(*training_reports)]
        std_vector = [np.std(i) for i in zip(*training_reports)]
        return mean_vector, std_vector

    def test_statistics(self, multiReport):
        """
        Gets a multiReport and calculates mean and standard deviation of the test_report.
        """
        _, test_reports, test_each = self.open_multiReport(multiReport)
        mean_vector = [np.mean(i) for i in zip(*test_reports)]
        std_vector = [np.std(i) for i in zip(*test_reports)]
        return mean_vector, std_vector, test_each[0]

    def training_statistics_sweep(self, multireports):
        feature_mean_vector = []
        feature_std_vector = []
        for report in range(len(multireports)):
            mean_vector, std_vector = self.training_statistics(multireports[report])
            feature_mean_vector.append(mean_vector)
            feature_std_vector.append(std_vector)
        meanofmeans = [np.mean(i) for i in zip(*feature_mean_vector)]
        stdofstd = [np.std(i) for i in zip(*feature_std_vector)]
        return meanofmeans, stdofstd

    def test_statistics_sweep(self, multireports):
        feature_mean_vector = []
        feature_std_vector = []
        for report in range(len(multireports)):
            mean_vector, std_vector, test_each = self.test_statistics(multireports[report])
            feature_mean_vector.append(mean_vector)
            feature_std_vector.append(std_vector)
        meanofmeans = [np.mean(i) for i in zip(*feature_mean_vector)]
        stdofstd = [np.std(i) for i in zip(*feature_std_vector)]
        return meanofmeans, stdofstd, test_each

###############################################################################
# Plotting funtions - single runs
###############################################################################

    def plot_training(self, save_path, training_report, run_num=None):
        """
        Make sure that the directory has been created by the FileManager.
        """
        plt.figure()
        x_data = np.arange(len(training_report))
        plt.plot(x_data, training_report,label = "DankAgent")
        plt.xlabel("Training episode")
        plt.ylabel("Total episode reward")
        plt.legend(loc='upper left')

        if run_num is None:
            self._save_plot(save_path, 'training_report')
        else:
            self._save_plot(save_path, 'training_report_run{}'.format(run_num))
        plt.close()

    def plot_test(self, save_path, test_report, run_num=None, testeach=None):
        """
        Make sure that the directory has been created by the FileManager.
        """
        plt.figure()
        if testeach is None:
            plt.plot(test_report, label = "DankAgent")
        else:
            x_data = testeach*(np.arange(len(test_report))+1)
            plt.plot(x_data, test_report, label = "DankAgent")
        plt.xlabel("Test episode")
        plt.ylabel("Average Test Reward")
        plt.legend(loc='upper left')
        if run_num is None:
            self._save_plot(save_path, 'test_report')
        else:
            self._save_plot(save_path, 'test_report_run{}'.format(run_num))
        plt.close()

###############################################################################
# Plotting funtions - multiple runs (with multiReport)
###############################################################################

    def plot_test_multireport(self, multiReport, save_path, name):
        """

        """
        _, test_reports, _ = self.open_multiReport(multiReport)
        mean_vector, std_vector, test_each = self.test_statistics(multiReport)

        # create_plot_test_mean_std

        plt.figure()
        x_data = test_each*(np.arange(len(mean_vector))+1)  # +1
        for run in range(len(multiReport)):
            plt.plot(x_data, test_reports[run], label = 'run {}'.format(run))
        plt.plot(x_data, mean_vector, label ='mean reward')
        plt.plot(x_data, np.add(mean_vector, std_vector), label='+STD', linestyle = '-.')
        plt.plot(x_data, np.subtract(mean_vector, std_vector), label='-STD', linestyle = '-.')
        plt.title("Test results for several runs")
        plt.xlabel("Intermediate test after training episode")
        plt.ylabel("Average reward")
        plt.legend(loc='upper left')
        self._save_plot(save_path, name)
        plt.close()

    def plot_training_multireport(self, multiReport, save_path, name):
        """

        """
        training_report, _, _ = self.open_multiReport(multiReport)
        mean_vector, std_vector = self.training_statistics(multiReport)

        plt.figure()
        x_data = np.arange(len(mean_vector))  # +1
        for run in range(len(multiReport)):
            plt.plot(x_data, training_report[run], label = 'run {}'.format(run))
        plt.plot(x_data, mean_vector, label ='mean of means')
        plt.plot(x_data, np.add(mean_vector, std_vector), label='+STD', linestyle = '-.')
        plt.plot(x_data, np.subtract(mean_vector, std_vector), label='-STD', linestyle = '-.')
        plt.title("Training results for several runs")
        plt.xlabel("Training episode")
        plt.ylabel("Episode reward")
        plt.legend(loc='upper left')
        self._save_plot(save_path, name)
        plt.close()

###############################################################################
# Plotting funtions - multiple features (with sweepReport)
###############################################################################

    def plot_training_sweep(self, sweepReport, save_path, name):
        multireports, feature_vector = self.open_sweepReport(sweepReport)
        meanofmeans, stdofstd = self.training_statistics_sweep(multireports)
        self.plot_mean_std_training(meanofmeans, stdofstd, feature_vector, save_path, name)
        self.create_sweep_plot_training(multireports, meanofmeans, stdofstd, feature_vector, save_path, name)

    def plot_mean_std_training(self, mean_vector, std_vector, feature_vector, save_path, name):
        plt.figure()
        x_data = np.arange(len(mean_vector))  # +1
        plt.plot(x_data, mean_vector, label ='mean of means')
        plt.plot(x_data, np.add(mean_vector, std_vector), label='+STD', linestyle = '-.')
        plt.plot(x_data, np.subtract(mean_vector, std_vector), label='-STD', linestyle = '-.')

        plt.title("Training results for several sweeps")
        plt.xlabel("Training episode")
        plt.ylabel("Episode reward")
        plt.legend(loc='upper left')
        self._save_plot(save_path, '{}_clean'.format(name))
        plt.close()

    def create_sweep_plot_training(self, multireports, mean_vector, std_vector, feature_vector, save_path, name):
        plt.figure()
        x_data = np.arange(len(mean_vector))  # +1
        for feature in range(len(multireports)):
            feature_mean, _ = self.training_statistics(multireports[feature])
            plt.plot(x_data, feature_mean, label = 'feature {}'.format(feature_vector[feature]))
        plt.plot(x_data, mean_vector, label ='mean of means')
        plt.plot(x_data, np.add(mean_vector, std_vector), label='+STD', linestyle = '-.')
        plt.plot(x_data, np.subtract(mean_vector, std_vector), label='-STD', linestyle = '-.')

        plt.title("Training results for several sweeps")
        plt.xlabel("Training episode")
        plt.ylabel("Episode reward")
        plt.legend(loc='upper left')
        self._save_plot(save_path, name)
        plt.close()


    def plot_test_sweep(self, sweepReport, save_path, name):
        multireports, feature_vector = self.open_sweepReport(sweepReport)
        meanofmeans, stdofstd, test_each = self.test_statistics_sweep(multireports)
        self.plot_mean_std_test(meanofmeans, stdofstd, feature_vector, test_each, save_path, name)
        self.create_sweep_plot_test(multireports, meanofmeans, stdofstd, feature_vector, test_each, save_path, name)

    def plot_mean_std_test(self, mean_vector, std_vector, feature_vector, test_each, save_path, name):
        plt.figure()
        x_data = test_each*(np.arange(len(mean_vector))+1)  # +1
        plt.plot(x_data, mean_vector, label ='sweep mean')
        plt.plot(x_data, np.add(mean_vector, std_vector), label='+STD', linestyle = '-.')
        plt.plot(x_data, np.subtract(mean_vector, std_vector), label='-STD', linestyle = '-.')

        plt.title("Test results for several sweeps")
        plt.xlabel("Training episode")
        plt.ylabel("Episode reward")
        plt.legend(loc='upper left')
        self._save_plot(save_path, '{}_clean'.format(name))
        plt.close()

    def create_sweep_plot_test(self, multireports, mean_vector, std_vector, feature_vector, test_each, save_path, name):
        plt.figure()
        x_data = test_each*(np.arange(len(mean_vector))+1)  # +1
        for feature in range(len(multireports)):
            feature_mean, _, _ = self.test_statistics(multireports[feature])
            plt.plot(x_data, feature_mean, label = 'feature {}'.format(feature_vector[feature]))
        plt.plot(x_data, mean_vector, label ='mean of means')
        plt.plot(x_data, np.add(mean_vector, std_vector), label='+STD', linestyle = '-.')
        plt.plot(x_data, np.subtract(mean_vector, std_vector), label='-STD', linestyle = '-.')

        plt.title("Test results for several sweeps")
        plt.xlabel("Training episode")
        plt.ylabel("Episode reward")
        plt.legend(loc='upper left')
        self._save_plot(save_path, name)
        plt.close()
