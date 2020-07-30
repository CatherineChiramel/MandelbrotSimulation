#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>



/*
	
	It represents a square area of the fractal to be rendered with the upper left point at x,y and lengths of size.
	it also contains information as to where to insert the calculated pixels as a starting shift (target_pos) and a size in pixels (target_size)
*/
typedef struct Block_{
	float x;
	float y;
	float size;
	int target_pos;
	int target_size;
} Block;



int checkMandelbrot(float real, float imag, int cutoff){
	 /*
		Task 2
		------
	
		The point (c = real + i*imag) is in the Mandelbrot set if the following sequence of complex numbers z_n is bounded
                (i.e. |z_n| <= 2  which means  sqrt(Real(z_n)^2 + Imag(z_n)^2) <=2 )

                 z_0 = 0
                 z_1 = z_0^2 + c
                 z_2 = z_1^2 + c
                
                 ... 

		 i.e. z_{n+1} = (z_n)^2 + c

                Consider that z_n are complex numbers!

                 
		Perform the iteration up to as many times as the cutoff states, at which point you assume the point is in the set.
		If the absolute value (|z_n|) is ever greater than 2, the series will diverge for sure and you can conclude that the point is not in the set.

                This function returns the number of iterations n after that sqrt(Re(z_n)^2 + Im(z_n)^2) > 2.


	*/
	int i, flag =0;
	int zReal_prev, zImg_prev, zReal_current, zImg_current;
	zReal_prev = 0;
	zImg_prev = 0;
	for (i=1;i<cutoff;i++)
	{
		zReal_current  = (zReal_prev*zReal_prev - zImg_prev * zImg_prev)+real;
		zImg_current = (2* zReal_prev * zImg_prev)+imag;
		if((zReal_current * zReal_current + zImg_current * zImg_current) > 4)
			{
				flag =1;
				break;
			}
		zReal_prev = zReal_current;
		zImg_prev = zImg_current;

	}

	if(flag == 1) return i; 
	return -1;                   

}



void HandleBlock(int my_rank, Block block, MPI_Win* window, int total_size_x, int**my_local_results, int max_number_iterations){

int v;

	for(v=0;v<block.target_size;++v){
		int b;
			for(b=0;b<block.target_size;++b){

				int result = checkMandelbrot(
					block.x + block.size * b / block.target_size
					,block.y + block.size * v / block.target_size
					, max_number_iterations
				);
				(*my_local_results)[b+v*block.target_size] = result; 
			}
	}


	for(v=0;v<block.target_size;++v){

		 /*
				- A triangular shape appears as the output
		*/
		MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, *window);
		MPI_Put((*my_local_results)+v*block.target_size, block.target_size ,MPI_INT, 0, (block.target_pos+ v*total_size_x), block.target_size, MPI_INT, *window);
		MPI_Win_unlock(0, *window);

	}
}




int main(int argc, char *argv[]){
	/*
		We render a square area with side lengths size (3rd command line parameter) with its upper left point given by the first two parameters.
		A fourth parameter sets the maximum number of iterations, to possibly bring out finer details.
	*/
	int output_size_pixels = 2000;
	float pos_x = -1.80f;
	float pos_y = -1.0f;
	float size = 2.0f;
	int max_number_iterations = 100;

	/*MPI seems to do fine with providing these to all participating processes*/
	if(argc >= 4){
		pos_x = atof(argv[1]);
		pos_y = atof(argv[2]);
		size = atof(argv[3]);
	}
	if(argc >= 5){
		max_number_iterations = atoi(argv[4]);
	}


 	static unsigned char white[3];
	white[0]=255; white[1]=255; white[2]=255;
 	static unsigned char black[3];
	black[0]=0; black[1]=0; black[2]=0;
 	static unsigned char greyscale[3];
	greyscale[0]=0; greyscale[1]=0; greyscale[2]=0;



	MPI_Init(&argc,&argv);
	int my_rank,total_ranks;
	MPI_Comm_size(MPI_COMM_WORLD,&total_ranks);
	MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);


	MPI_Win window;
	int * shared_data;
	/*MPI_Datatype Blocktype;
	int len[2]= {3,2};
	int disp[3];
	disp[0]=0;
	disp[1]= 3*sizeof(float);
	MPI_Type_create_struct(3,len,disp,&Blocktype);
	MPI_Type_commit(&Blocktype);*/

	
	

	if(my_rank == 0){
		/*Allocate memory and make available to others*/

		MPI_Aint siz = output_size_pixels * output_size_pixels * sizeof(int);


		MPI_Alloc_mem(siz, MPI_INFO_NULL, &shared_data);
		MPI_Win_create(shared_data, siz, sizeof(int),MPI_INFO_NULL, MPI_COMM_WORLD, &window);


	} else {
		
		MPI_Win_create(shared_data, 0,sizeof(int),MPI_INFO_NULL, MPI_COMM_WORLD, &window);
	}


	int * my_result_vals;

	int blocks_per_direction = 1;
	int total_blocks = blocks_per_direction * blocks_per_direction;
	int block_pixel_size = output_size_pixels / blocks_per_direction;
	
	my_result_vals = malloc(sizeof(int)*(block_pixel_size  * block_pixel_size));



	if(my_rank == 1){
		int block_coord_x = 0; 
		int block_coord_y = 0;
		
		Block block;
		block.x = pos_x + size / blocks_per_direction * block_coord_x;
		block.y = pos_y + size / blocks_per_direction * block_coord_y;
		block.size = size / blocks_per_direction;

		block.target_size = block_pixel_size;
		/*Start of the first row in our target matrix.
		Second row will start with an offset of +output_size_pixels, etc.*/
		block.target_pos = block.target_size * (block_coord_x + block_coord_y * output_size_pixels);


		HandleBlock(my_rank,block,&window,output_size_pixels, &my_result_vals, max_number_iterations);
	}


	/*Before outputting the result we wait for all the values*/
	MPI_Barrier(MPI_COMM_WORLD);

	/*Temporary storage no longer needed*/
	free(my_result_vals);
    int i,b;

	if(my_rank == 0){
		char filename[100];
		sprintf(filename,"Mandelbrot_x%f y%f size %f.ppm",pos_x,pos_y,size);
		FILE *fp = fopen(filename, "wb"); /* b - binary mode */
		fprintf(fp, "P6\n%d %d\n255\n", output_size_pixels,output_size_pixels);



		for(i=0;i<output_size_pixels;++i){
			for(b=0;b<output_size_pixels;++b){
				int res = shared_data[i*output_size_pixels + b];
				if( res > 0){
					greyscale[0]=250-200*res/max_number_iterations;
					greyscale[1]=250-200*res/max_number_iterations;
					greyscale[2]=250-200*res/max_number_iterations;
					fwrite(greyscale, 1 , 3 , fp);
				} else {
					fwrite(black, 1 , 3 , fp);
				}

			}
		}

		fclose(fp);
	}





	MPI_Win_free(&window);
	MPI_Finalize();

	return 0;
}
