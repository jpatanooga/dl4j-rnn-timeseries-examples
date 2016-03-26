package org.deeplearning4j.examples.rnn.strata.physionet.output.single;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.net.URL;
import java.text.DecimalFormat;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Random;

import org.apache.commons.compress.archivers.tar.TarArchiveEntry;
import org.apache.commons.compress.archivers.tar.TarArchiveInputStream;
import org.apache.commons.compress.compressors.gzip.GzipCompressorInputStream;
import org.apache.commons.io.FileUtils;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.examples.rnn.strata.physionet.output.single.PhysioNet_ICU_SingleLabel_Iterator;
import org.deeplearning4j.examples.rnn.strata.physionet.utils.EvalScoreTracker;
import org.deeplearning4j.examples.rnn.strata.physionet.utils.PhysioNetDataUtils;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

public class PhysioNet_LSTM_Model {

	
	
	public static void main( String[] args ) throws Exception {
		
		trainPhysioNetExample();
		
	}
	
	public static void scoreInputWithModel(String modelPath) throws Exception {
		
		
	}
	
	public static void resumeTrainingPhysioNetModel(String modelPath) throws Exception {
		
		
		
		
	}
	
	public static void trainPhysioNetExample() throws Exception {
		
		String existingModelPath = null; //"/tmp/rnns/physionet/models/dl4j_model_run_2016-03-21_11_27_18/epoch_9_f1_0.7380/";
		
		int lstmLayerSize = 300;					//Number of units in each GravesLSTM layer
		int miniBatchSize = 20;						//Size of mini batch to use when  training
		//int totalExamplesToTrainWith = 1100;
		
		int trainingExamples = 200; // 2800;
		int testExamples = 100; //600;
		int validateExamples = 100; //600;
		
		double learningRate = 0.009;
		
		int numEpochs = 5;							//Total number of training + sample generation epochs
		Random rng = new Random(12345);
		SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd_HH_mm_ss");
		
		int nOut = 1; //iter.totalOutcomes();
		
		
		PhysioNet_ICU_SingleLabel_Iterator iter = new PhysioNet_ICU_SingleLabel_Iterator( "/tmp/set-a-full-splits-1/train/", "src/test/resources/physionet_schema_zmzuv_0.txt", "src/test/resources/data/physionet/sample/set-a-labels/Outcomes-a.txt", miniBatchSize, trainingExamples);
		
		
		PhysioNet_ICU_SingleLabel_Iterator iter_validate = new PhysioNet_ICU_SingleLabel_Iterator( "/tmp/set-a-full-splits-1/validate/", "src/test/resources/physionet_schema_zmzuv_0.txt", "src/test/resources/data/physionet/sample/set-a-labels/Outcomes-a.txt", 1, validateExamples);
		
		PhysioNet_ICU_SingleLabel_Iterator test_iter = new PhysioNet_ICU_SingleLabel_Iterator( "/tmp/set-a-full-splits-1/test/", "src/test/resources/physionet_schema_zmzuv_0.txt", "src/test/resources/data/physionet/sample/set-a-labels/Outcomes-a.txt", testExamples, testExamples);
		
		iter.reset();
		test_iter.reset();
		iter_validate.reset();
		
		System.out.println( "We have " + iter.inputColumns() + " input columns." );
		System.out.println( "We have " + nOut + " output columns." );
		
		// *****************************
		// TODO: Drop:
/*
dropout for rnns is applied on the input activations only, not recurrent activations
as is common in the literature
same as other layers
so .dropout(0.5) with .regularization(true)		
 */
		
		//Set up network configuration:
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
			.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
			.learningRate( learningRate )
			.rmsDecay(0.95)
			.seed(12345)
			.regularization(true)
			.l2(0.001)
			//.dropOut(0.5)
			.list(3)
			.layer(0, new GravesLSTM.Builder().nIn(iter.inputColumns()).nOut(lstmLayerSize)
					.updater(Updater.RMSPROP)
					.activation("tanh").weightInit(WeightInit.DISTRIBUTION)
					.dist(new UniformDistribution(-0.08, 0.08)).build())
			.layer(1, new GravesLSTM.Builder().nIn(lstmLayerSize).nOut(lstmLayerSize)
					.updater(Updater.RMSPROP)
					.activation("tanh").weightInit(WeightInit.DISTRIBUTION)
					.dist(new UniformDistribution(-0.08, 0.08)).build())
			.layer(2, new RnnOutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD).activation("sigmoid")        //MCXENT + softmax for classification
					.updater(Updater.RMSPROP)
					.nIn(lstmLayerSize).nOut(nOut).weightInit(WeightInit.DISTRIBUTION)
					.dist(new UniformDistribution(-0.08, 0.08)).build())
			.pretrain(false).backprop(true)
			.build();
		

		

		
		MultiLayerNetwork net = new MultiLayerNetwork(conf);
		net.init();
		net.setListeners(new ScoreIterationListener(1));

// DAVE: UNCOMMENT HERE AND REPLACE DIRS TO RESUME TRAINING...
//		System.out.println( "Loading old parameters [test] >> " );
		
		if (null != existingModelPath) {
		
			PhysioNetDataUtils.loadDL4JNetworkParameters( net, existingModelPath );

		
			
			iter_validate.reset();
			
			Evaluation evaluation_validate = new Evaluation( nOut );
            while(iter_validate.hasNext()){
                DataSet t = iter_validate.next();
                INDArray features = t.getFeatureMatrix();
                INDArray lables = t.getLabels();
                INDArray inMask = t.getFeaturesMaskArray();
                INDArray outMask = t.getLabelsMaskArray();
                INDArray predicted = net.output(features,false,inMask,outMask);

                evaluation_validate.evalTimeSeries(lables,predicted,outMask);
                
            }
            System.out.println( "\nParameter Load --- Pre Check: Validate Evaluation: ");
            System.out.println( evaluation_validate.stats() );			
			
		}
		
		org.deeplearning4j.eval.Evaluation e = new Evaluation();
		
		
		//Print the  number of parameters in the network (and for each layer)
		Layer[] layers = net.getLayers();
		int totalNumParams = 0;
		for( int i=0; i<layers.length; i++ ){
			int nParams = layers[i].numParams();
			System.out.println("Number of parameters in layer " + i + ": " + nParams);
			totalNumParams += nParams;
		}
		System.out.println("Total number of network parameters: " + totalNumParams);
		
		
		long startTime = System.currentTimeMillis();

		
		EvalScoreTracker f1Tracker = new EvalScoreTracker( 20 );
        Date now = new Date();
        String strDate = sdf.format(now);        
        DecimalFormat df = new DecimalFormat("#.0000"); 
        String runID = "dl4j_model_run_" + strDate;
		
		
		//Do training, and then generate and print samples from network
		for ( int i=0; i<numEpochs; i++ ){
			
			iter.reset();
			test_iter.reset();
			iter_validate.reset();
			
			
			net.fit(iter);
			
			System.out.println("--------------------");
			System.out.println("Completed epoch " + i );
			
			long curTime = System.currentTimeMillis();
			
			long progressElapsedTimeMS = (curTime - startTime);
			long processElpasedTimeMin = progressElapsedTimeMS / 1000 / 60;
			
			System.out.println("Elapsed Time So Far: " + processElpasedTimeMin + " minutes");
			
			//Evaluation eval = new Evaluation( 2 );
		
			//INDArray output = net.output( testInput );
			
			iter.reset();
			test_iter.reset();
			iter_validate.reset();
			
			Evaluation evaluation_train = new Evaluation( nOut );
            while(iter.hasNext()){
                DataSet t = iter.next();
                INDArray features = t.getFeatureMatrix();
                INDArray lables = t.getLabels();
                INDArray inMask = t.getFeaturesMaskArray();
                INDArray outMask = t.getLabelsMaskArray();
                INDArray predicted = net.output(features,false,inMask,outMask);

                evaluation_train.evalTimeSeries(lables,predicted,outMask);
                
            }			
			
			Evaluation evaluation_validate = new Evaluation( nOut );
            while(iter_validate.hasNext()){
                DataSet t = iter_validate.next();
                INDArray features = t.getFeatureMatrix();
                INDArray lables = t.getLabels();
                INDArray inMask = t.getFeaturesMaskArray();
                INDArray outMask = t.getLabelsMaskArray();
                INDArray predicted = net.output(features,false,inMask,outMask);
                /*
                System.out.println("predicted: ");
                System.out.println( predicted.getRow(0) );
                System.out.println("label: ");
                System.out.println( lables.getRow(0) );
*/
                evaluation_validate.evalTimeSeries(lables,predicted,outMask);
                //evaluation_validate.ev
                
            }
            //test_iter.reset();
            System.out.println( "\nTrain Evaluation: ");
            System.out.println( evaluation_train.stats() );
            System.out.println( "\nValidate Evaluation: ");
            System.out.println( evaluation_validate.stats() );
            
            f1Tracker.addF1( i, evaluation_train.f1(), evaluation_validate.f1() );
            f1Tracker.printWindow();
            
            String epochID = "epoch_" + i + "_f1_0" + df.format( evaluation_train.f1() );
            
            String fileNamePathBase = "/tmp/rnns/physionet/models/" + runID + "/" + epochID + "/";
            
            File dirs = new File(fileNamePathBase);
            dirs.mkdirs();
            
            PhysioNetDataUtils.saveDL4JNetwork( net, fileNamePathBase );
            
		}
		
/*
also fyi for evaluation (using Evaluation class), you can just pass the full/padded output along with the label mask array, and it'll do the subsetting for you
i.e., it'll only do the evaluation where the real data is (according to mask array) and ignore the padded time steps
also you can do an iamax op along dimension 1 for the label mask array to work out where real output is
that only works for the many-to-one case though		
 */
		
		
		Evaluation evaluation_final_test = new Evaluation(2);
        while(test_iter.hasNext()){
            DataSet t = test_iter.next();
            INDArray features = t.getFeatureMatrix();
            INDArray lables = t.getLabels();
            INDArray inMask = t.getFeaturesMaskArray();
            INDArray outMask = t.getLabelsMaskArray();
            INDArray predicted = net.output(features,false,inMask,outMask);

            evaluation_final_test.evalTimeSeries(lables,predicted,outMask);
            
        }
        //test_iter.reset();
        System.out.println( "\n\n\nFinal Test Evaluation: ");
        System.out.println( evaluation_final_test.stats() );
		
		
		System.out.println("\n\nExample complete");
		
		long endTime = System.currentTimeMillis();
		
		long elapsedTimeMS = (endTime - startTime);
		long elpasedTimeSeconds = elapsedTimeMS / 1000;
		long elapsedTimeMinutes = elpasedTimeSeconds / 60;
		long elapsedTimeHours = elapsedTimeMinutes / 60;

		System.out.println("Training took " + (elpasedTimeSeconds) + " seconds");
		System.out.println("Training took " + elapsedTimeMinutes + " minutes");
		System.out.println("Training took " + elapsedTimeHours + " hours");
		
	}



	

	
}
