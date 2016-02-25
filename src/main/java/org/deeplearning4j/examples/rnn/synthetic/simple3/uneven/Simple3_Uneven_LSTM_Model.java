package org.deeplearning4j.examples.rnn.synthetic.simple3.uneven;

import java.util.Random;

import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.examples.rnn.synthetic.simple2.Simple2Dataset_Iterator;
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

public class Simple3_Uneven_LSTM_Model {



	public static void main( String[] args ) throws Exception {
		
		trainExample();
		
	}
	
	public static void trainExample() throws Exception {
		
		int lstmLayerSize = 200;					//Number of units in each GravesLSTM layer
		int miniBatchSize = 10;						//Size of mini batch to use when  training
		int totalExamplesToTrainWith = 40;
		//int examplesPerEpoch = 50 * miniBatchSize;	//i.e., how many examples to learn on between generating samples
		//int exampleLength = 100;					//Length of each training example
		int numEpochs = 50;							//Total number of training + sample generation epochs
		//int nSamplesToGenerate = 4;					//Number of samples to generate after each training epoch
		//int nCharactersToSample = 300;				//Length of each sample to generate
		//String generationInitialization = null;		//Optional character initialization; a random character is used if null
		// Above is Used to 'prime' the LSTM with a character sequence to continue/complete.
		// Initialization characters must all be in CharacterIterator.getMinimalCharacterSet() by default
		Random rng = new Random(12345);
		
		//Get a DataSetIterator that handles vectorization of text into something we can use to train
		// our GravesLSTM network.
		//CharacterIterator iter = getShakespeareIterator(miniBatchSize,exampleLength,examplesPerEpoch);
		int nOut = 2; //iter.totalOutcomes();
		
		//SyntheticDataIterator iter = getPhysioNetIterator( miniBatchSize, totalExamplesToTrainWith );
		//Simple3_Uneven_Dataset_Iterator iter = new Simple3_Uneven_Dataset_Iterator("src/test/resources/data/synthetic/simple_2/simple_2_data.txt", "src/test/resources/data/synthetic/simple_2/simple_2_labels.txt", miniBatchSize, totalExamplesToTrainWith);
		Simple3_Uneven_Dataset_Iterator iter = new Simple3_Uneven_Dataset_Iterator("src/test/resources/data/synthetic/simple_3_uneven/simple_3_uneven_data.txt", "src/test/resources/data/synthetic/simple_3_uneven/simple_3_uneven_labels.txt", miniBatchSize, totalExamplesToTrainWith, 4 );
		
		//SyntheticDataIterator test_iter = getPhysioNetIterator( miniBatchSize, 1000 );
		//Simple3_Uneven_Dataset_Iterator test_iter = new Simple3_Uneven_Dataset_Iterator("src/test/resources/data/synthetic/simple_2/simple_2_data.txt", "src/test/resources/data/synthetic/simple_2/simple_2_labels.txt", miniBatchSize, totalExamplesToTrainWith);
		Simple3_Uneven_Dataset_Iterator test_iter = new Simple3_Uneven_Dataset_Iterator("src/test/resources/data/synthetic/simple_3_uneven/simple_3_uneven_data.txt", "src/test/resources/data/synthetic/simple_3_uneven/simple_3_uneven_labels.txt", totalExamplesToTrainWith, totalExamplesToTrainWith, 4 );
		
		iter.reset();
		
		System.out.println( "We have " + iter.inputColumns() + " input columns." );
		
		//Set up network configuration:
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
			.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
			.learningRate(0.005)
			.rmsDecay(0.95)
			.seed(12345)
			.regularization(true)
			.l2(0.001)
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
		
		//Print the  number of parameters in the network (and for each layer)
		Layer[] layers = net.getLayers();
		int totalNumParams = 0;
		for( int i=0; i<layers.length; i++ ){
			int nParams = layers[i].numParams();
			System.out.println("Number of parameters in layer " + i + ": " + nParams);
			totalNumParams += nParams;
		}
		System.out.println("Total number of network parameters: " + totalNumParams);
		
		
		//Do training, and then generate and print samples from network
		for( int i=0; i<numEpochs; i++ ){
			net.fit(iter);
			
			
			System.out.println("--------------------");
			System.out.println("Completed epoch " + i );
			//System.out.println("Sampling characters from network given initialization \""+ (generationInitialization == null ? "" : generationInitialization) +"\"");
			//String[] samples = sampleCharactersFromNetwork(generationInitialization,net,iter,rng,nCharactersToSample,nSamplesToGenerate);
			//for( int j=0; j<samples.length; j++ ){
			//	System.out.println("----- Sample " + j + " -----");
			//	System.out.println(samples[j]);
			//	System.out.println();
			//}
			
			iter.reset();	//Reset iterator for another epoch
			
		}
			
			
			//Evaluation eval = new Evaluation( 2 );
		
			//INDArray output = net.output( testInput );
			
			Evaluation evaluation = new Evaluation(2);
            while(test_iter.hasNext()){
                DataSet t = test_iter.next();
                INDArray features = t.getFeatureMatrix();
                INDArray lables = t.getLabels();
                INDArray inMask = t.getFeaturesMaskArray();
                INDArray outMask = t.getLabelsMaskArray();
                INDArray predicted = net.output(features,false,inMask,outMask);

                evaluation.evalTimeSeries(lables,predicted,outMask);
                
            }
            System.out.println( evaluation.stats() );
            test_iter.reset();
			
			
		//}
		
/*
also fyi for evaluation (using Evaluation class), you can just pass the full/padded output along with the label mask array, and it'll do the subsetting for you
i.e., it'll only do the evaluation where the real data is (according to mask array) and ignore the padded time steps
also you can do an iamax op along dimension 1 for the label mask array to work out where real output is
that only works for the many-to-one case though		
 */
		
		
		System.out.println("\n\nExample complete");
	}			
	
}
