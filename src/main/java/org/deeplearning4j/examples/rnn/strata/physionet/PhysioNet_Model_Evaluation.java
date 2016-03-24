package org.deeplearning4j.examples.rnn.strata.physionet;

import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Random;

import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.examples.rnn.strata.physionet.utils.PhysioNetDataUtils;
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

public class PhysioNet_Model_Evaluation {

	public static void main( String[] args ) throws Exception {

		evaluateExistingModel();
		
	}
	
	public static void evaluateExistingModel() throws IOException {
		
		String modelPath = "/tmp/rnns/physionet/models/dl4j_model_run_2016-03-20_17_21_08/epoch_9_f1_0.8684/";

		int lstmLayerSize = 300;					//Number of units in each GravesLSTM layer
		int miniBatchSize = 20;						//Size of mini batch to use when  training
		//int totalExamplesToTrainWith = 1100;
		
		int trainingExamples = 2800;
		int testExamples = 600;
		int validateExamples = 600;
		
		double learningRate = 0.009;
		
		int numEpochs = 10;							//Total number of training + sample generation epochs
		Random rng = new Random(12345);
		SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd_HH_mm_ss");
		
		int nOut = 2; //iter.totalOutcomes();
		
		//PhysioNet_ICU_Mortality_Iterator iter = getPhysioNetIterator( miniBatchSize, totalExamplesToTrainWith );
		
		PhysioNet_ICU_Mortality_Iterator iter = new PhysioNet_ICU_Mortality_Iterator( "/tmp/set-a-full-splits-1/train/", "src/test/resources/physionet_schema_zmzuv_0.txt", "src/test/resources/data/physionet/sample/set-a-labels/Outcomes-a.txt", miniBatchSize, trainingExamples);
		
		
		PhysioNet_ICU_Mortality_Iterator iter_validate = new PhysioNet_ICU_Mortality_Iterator( "/tmp/set-a-full-splits-1/validate/", "src/test/resources/physionet_schema_zmzuv_0.txt", "src/test/resources/data/physionet/sample/set-a-labels/Outcomes-a.txt", validateExamples, validateExamples);
		
	//	PhysioNet_ICU_Mortality_Iterator test_iter = getPhysioNetIterator( miniBatchSize, 100 );
		
		//PhysioNet_ICU_Mortality_Iterator test_iter = new PhysioNet_ICU_Mortality_Iterator( "/tmp/set-a-balanced-5/test/", "src/test/resources/physionet_schema.txt", "src/test/resources/data/physionet/sample/set-a-labels/Outcomes-a.txt", miniBatchSize, 20);
		PhysioNet_ICU_Mortality_Iterator test_iter = new PhysioNet_ICU_Mortality_Iterator( "/tmp/set-a-full-splits-1/test/", "src/test/resources/physionet_schema_zmzuv_0.txt", "src/test/resources/data/physionet/sample/set-a-labels/Outcomes-a.txt", testExamples, testExamples);
		
		iter.reset();
		test_iter.reset();
		iter_validate.reset();
		
		System.out.println( "We have " + iter.inputColumns() + " input columns." );
		
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
			.layer(2, new RnnOutputLayer.Builder(LossFunction.MCXENT).activation("softmax")        //MCXENT + softmax for classification
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
		PhysioNetDataUtils.loadDL4JNetworkParameters( net, modelPath );


			
			iter_validate.reset();
			
			Evaluation evaluation_validate = new Evaluation(2);
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
            
		
	}
	
}
