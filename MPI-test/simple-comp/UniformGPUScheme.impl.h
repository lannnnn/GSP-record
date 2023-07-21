//======================================================================================================================
//
//  This file is part of waLBerla. waLBerla is free software: you can
//  redistribute it and/or modify it under the terms of the GNU General Public
//  License as published by the Free Software Foundation, either version 3 of
//  the License, or (at your option) any later version.
//
//  waLBerla is distributed in the hope that it will be useful, but WITHOUT
//  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
//  FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
//  for more details.
//
//  You should have received a copy of the GNU General Public License along
//  with waLBerla (see COPYING.txt). If not, see <http://www.gnu.org/licenses/>.
//
//! \file UniformGPUScheme.impl.h
//! \ingroup cuda
//! \author Martin Bauer <martin.bauer@fau.de>
//
//======================================================================================================================

#include "cuda/ParallelStreams.h"
#include <cuZFP.h>

namespace walberla {
namespace cuda {
namespace communication {


   template<typename Stencil>
   UniformGPUScheme<Stencil>::UniformGPUScheme( weak_ptr <StructuredBlockForest> bf,
                                                bool sendDirectlyFromGPU,
                                                const int tag )
        : blockForest_( bf ),
          setupBeforeNextCommunication_( true ),
          communicationInProgress_( false ),
          sendFromGPU_( sendDirectlyFromGPU ),
          bufferSystemCPU_( mpi::MPIManager::instance()->comm(), tag ),
          bufferSystemGPU_( mpi::MPIManager::instance()->comm(), tag ),
	  bufferSystemGPU_copy( mpi::MPIManager::instance()->comm(), tag ),
          parallelSectionManager_( -1 ),
          requiredBlockSelectors_( Set<SUID>::emptySet() ),
          incompatibleBlockSelectors_( Set<SUID>::emptySet() )
   {}

   template<typename Stencil>
   UniformGPUScheme<Stencil>::UniformGPUScheme( weak_ptr <StructuredBlockForest> bf,
                                                const Set<SUID> & requiredBlockSelectors,
                                                const Set<SUID> & incompatibleBlockSelectors,
                                                bool sendDirectlyFromGPU,
                                                const int tag )
      : blockForest_( bf ),
        setupBeforeNextCommunication_( true ),
        communicationInProgress_( false ),
        sendFromGPU_( sendDirectlyFromGPU ),
        bufferSystemCPU_( mpi::MPIManager::instance()->comm(), tag ),
        bufferSystemGPU_( mpi::MPIManager::instance()->comm(), tag ),
	bufferSystemGPU_copy( mpi::MPIManager::instance()->comm(), tag ),
        parallelSectionManager_( -1 ),
        requiredBlockSelectors_( requiredBlockSelectors ),
        incompatibleBlockSelectors_( incompatibleBlockSelectors )
   {}


   template<typename Stencil>
   void UniformGPUScheme<Stencil>::startCommunication( cudaStream_t stream )
   {
      WALBERLA_ASSERT( !communicationInProgress_ )
      auto forest = blockForest_.lock();

      auto currentBlockForestStamp = forest->getBlockForest().getModificationStamp();
      if( setupBeforeNextCommunication_ || currentBlockForestStamp != forestModificationStamp_ )
         setupCommunication();

      // Schedule Receives
      if( sendFromGPU_ )
         //bufferSystemGPU_.scheduleReceives();
	 bufferSystemGPU_copy.scheduleReceives();
      else
         bufferSystemCPU_.scheduleReceives();
	 

      if( !sendFromGPU_ )
         for( auto it : headers_ )
            //bufferSystemGPU_.sendBuffer( it.first ).clear();
	    bufferSystemGPU_copy.sendBuffer( it.first ).clear();

      //size_t ori_size = 0; //count size
      //size_t comp_size = 0;
      // Start filling send buffers
      {
         auto parallelSection = parallelSectionManager_.parallelSection( stream );
         for( auto &iBlock : *forest )
         {
            auto block = dynamic_cast< Block * >( &iBlock );
	    
            if( !selectable::isSetSelected( block->getState(), requiredBlockSelectors_, incompatibleBlockSelectors_ ) )
               continue;

            for( auto dir = Stencil::beginNoCenter(); dir != Stencil::end(); ++dir )
            {
               const auto neighborIdx = blockforest::getBlockNeighborhoodSectionIndex( *dir );
               if( block->getNeighborhoodSectionSize( neighborIdx ) == uint_t( 0 ))
                  continue;
               auto nProcess = mpi::MPIRank( block->getNeighborProcess( neighborIdx, uint_t( 0 )));

               if( !selectable::isSetSelected( block->getNeighborState( neighborIdx, uint_t(0) ), requiredBlockSelectors_, incompatibleBlockSelectors_ ) )
                  continue;

               for( auto &pi : packInfos_ )
               {
                  parallelSection.run([&](auto s) {
                     auto size = pi->size( *dir, block );
                     //auto gpuDataPtr = bufferSystemGPU_.sendBuffer( nProcess ).advanceNoResize( size );
		     unsigned char* gpuDataPtr = NULL;
		     //WALBERLA_ASSERT_NOT_NULLPTR( gpuDataPtr_comp )
		     cudaMalloc((void**)&gpuDataPtr, size);		     
		     pi->pack( *dir, gpuDataPtr, block, s );
		     
		     //compress the data
		     zfp_stream* zfp;
		     bitstream* bstream;
		     zfp_field *in_field = zfp_field_1d(reinterpret_cast<double*>(gpuDataPtr), zfp_type_double, size/sizeof(double));

		     int rate = 32;
		     int dims = 1;  
		     zfp = zfp_stream_open(NULL);
		     zfp_stream_set_rate(zfp, rate, in_field->type, dims, zfp_false);
		     size_t buffsize = zfp_stream_maximum_size(zfp, in_field);
		     auto gpuDataPtr_comp = bufferSystemGPU_copy.sendBuffer( nProcess ).advanceNoResize(buffsize);
		     //ori_size += size;
		     //comp_size += buffsize;
		     WALBERLA_ASSERT_NOT_NULLPTR( gpuDataPtr_comp )
		     bstream = stream_open(reinterpret_cast<double*>(gpuDataPtr_comp), buffsize);
		     zfp_stream_set_bit_stream(zfp, bstream);
		     if (zfp_stream_set_execution(zfp, zfp_exec_cuda)) {
		     	cuda_compress(zfp, in_field);
		     } else {
			fprintf(stderr, "Failed to compress");
		     }	
		     stream_close(bstream);
		     zfp_field_free(in_field);
		     zfp_stream_close(zfp);
		     cudaFree(gpuDataPtr);
		     //fprintf(stderr, "well compress\n");
		     gpuDataPtr_comp = reinterpret_cast<unsigned char*>(gpuDataPtr_comp);
		     //fprintf(stderr, "well cast\n");
		     //fprintf(stderr, "gpuDataPtr size: %ld, compress size: %ld\n", size, buffsize);

                     if( !sendFromGPU_ )
                     {
                        auto cpuDataPtr = bufferSystemCPU_.sendBuffer( nProcess ).advanceNoResize( size );
                        WALBERLA_ASSERT_NOT_NULLPTR( cpuDataPtr )
                        WALBERLA_CUDA_CHECK( cudaMemcpyAsync( cpuDataPtr, gpuDataPtr, size, cudaMemcpyDeviceToHost, s ))
                     }
                  });
               }
            }
         }
      }

      // wait for packing to finish
      cudaStreamSynchronize( stream );
      //fprintf(stderr, "original MPI communication data size: %ld, compressed data size: %ld\n", ori_size, comp_size);
      if( sendFromGPU_ ) {
         //bufferSystemGPU_.sendAll();
	 //fprintf(stderr, "before\n");
	 bufferSystemGPU_copy.sendAll();
	 //fprintf(stderr, "after\n");
      }
      else
         bufferSystemCPU_.sendAll();
      communicationInProgress_ = true;
   }


   template<typename Stencil>
   void UniformGPUScheme<Stencil>::wait( cudaStream_t stream )
   {
      WALBERLA_ASSERT( communicationInProgress_ )

      auto forest = blockForest_.lock();

      if( sendFromGPU_ )
      {
         auto parallelSection = parallelSectionManager_.parallelSection( stream );
         //for( auto recvInfo = bufferSystemGPU_.begin(); recvInfo != bufferSystemGPU_.end(); ++recvInfo )
         for( auto recvInfo = bufferSystemGPU_copy.begin(); recvInfo != bufferSystemGPU_.end(); ++recvInfo )
	 {
            recvInfo.buffer().clear();
            for( auto &header : headers_[recvInfo.rank()] )
            {
               auto block = dynamic_cast< Block * >( forest->getBlock( header.blockId ));

               for( auto &pi : packInfos_ )
               {
                  auto size = pi->size( header.dir, block );
                  //auto gpuDataPtr_comp = reinterpret_cast<double*>(recvInfo.buffer().advanceNoResize(size*1/2+32));
		  double* gpuDataPtr = NULL;
		  cudaMalloc((void**)&gpuDataPtr, size);
                  //WALBERLA_ASSERT_NOT_NULLPTR( gpuDataPtr_comp )

		  //decompress
		  zfp_stream* zfp;
		  bitstream* bstream;
		  zfp_field *out_field = zfp_field_1d(gpuDataPtr, zfp_type_double, size/sizeof(double));

	          int rate = 32;
		  int dims = 1;
		  zfp = zfp_stream_open(NULL);
		  zfp_stream_set_rate(zfp, rate, out_field->type, dims, zfp_false);
		  size_t buffsize = zfp_stream_maximum_size(zfp, out_field);
		  auto gpuDataPtr_comp = reinterpret_cast<double*>(recvInfo.buffer().advanceNoResize(buffsize));
		  WALBERLA_ASSERT_NOT_NULLPTR( gpuDataPtr_comp );
		  bstream = stream_open(gpuDataPtr_comp, buffsize);
		  zfp_stream_set_bit_stream(zfp, bstream);
		
		  if (zfp_stream_set_execution(zfp, zfp_exec_cuda)) {                 
			cuda_decompress(zfp, out_field);         
		  } else {
			fprintf(stderr, "Failed to decompress");
		  }
		  stream_close(bstream);
		  zfp_field_free(out_field);
		  zfp_stream_close(zfp);
                  parallelSection.run([&](auto s) {
                     pi->unpack( stencil::inverseDir[header.dir], reinterpret_cast<unsigned char*>(gpuDataPtr), block, s );
                  });
		  cudaFree(gpuDataPtr);
               }
            }
         }
      }
      else
      {
         auto parallelSection = parallelSectionManager_.parallelSection( stream );
         for( auto recvInfo = bufferSystemCPU_.begin(); recvInfo != bufferSystemCPU_.end(); ++recvInfo )
         {
            //auto &gpuBuffer = bufferSystemGPU_.sendBuffer( recvInfo.rank());
	    auto &gpuBuffer = bufferSystemGPU_copy.sendBuffer( recvInfo.rank());

            recvInfo.buffer().clear();
            gpuBuffer.clear();
            for( auto &header : headers_[recvInfo.rank()] ) {
               auto block = dynamic_cast< Block * >( forest->getBlock( header.blockId ));

               for( auto &pi : packInfos_ )
               {
                  auto size = pi->size( header.dir, block );
                  auto cpuDataPtr = recvInfo.buffer().advanceNoResize( size );
                  auto gpuDataPtr = gpuBuffer.advanceNoResize( size );
                  WALBERLA_ASSERT_NOT_NULLPTR( cpuDataPtr )
                  WALBERLA_ASSERT_NOT_NULLPTR( gpuDataPtr )

                  parallelSection.run([&](auto s) {
                     WALBERLA_CUDA_CHECK( cudaMemcpyAsync( gpuDataPtr, cpuDataPtr, size,
                                                           cudaMemcpyHostToDevice, s ))
                     pi->unpack( stencil::inverseDir[header.dir], gpuDataPtr, block, s );
                  });
               }
            }
         }
      }
      communicationInProgress_ = false;
   }


   template<typename Stencil>
   void UniformGPUScheme<Stencil>::setupCommunication()
   {
      auto forest = blockForest_.lock();

      headers_.clear();

      std::map<mpi::MPIRank, mpi::MPISize> receiverInfo; // how many bytes to send to each neighbor

      mpi::BufferSystem headerExchangeBs( mpi::MPIManager::instance()->comm(), 123 );
      
      for( auto &iBlock : *forest ) {
         auto block = dynamic_cast< Block * >( &iBlock );

         if( !selectable::isSetSelected( block->getState(), requiredBlockSelectors_, incompatibleBlockSelectors_ ) )
            continue;

         for( auto dir = Stencil::beginNoCenter(); dir != Stencil::end(); ++dir ) {
            // skip if block has no neighbors in this direction
            const auto neighborIdx = blockforest::getBlockNeighborhoodSectionIndex( *dir );
            if( block->getNeighborhoodSectionSize( neighborIdx ) == uint_t( 0 ))
               continue;

            WALBERLA_ASSERT( block->neighborhoodSectionHasEquallySizedBlock( neighborIdx ),
                             "Works for uniform setups only" )
            WALBERLA_ASSERT_EQUAL( block->getNeighborhoodSectionSize( neighborIdx ), uint_t( 1 ),
                                   "Works for uniform setups only" )

            const BlockID &nBlockId = block->getNeighborId( neighborIdx, uint_t( 0 ));

            if( !selectable::isSetSelected( block->getNeighborState( neighborIdx, uint_t(0) ), requiredBlockSelectors_, incompatibleBlockSelectors_ ) )
               continue;

            auto nProcess = mpi::MPIRank( block->getNeighborProcess( neighborIdx, uint_t( 0 )));

	    zfp_stream* zfp = zfp_stream_open(NULL);
	    zfp_stream_set_rate(zfp, 32, zfp_type_double, 1, zfp_false);
	    //size_t buffsize = zfp_stream_maximum_size(zfp, zfp_field_1d(NULL, zfp_type_double, pi->size( *dir, block ))/sizeof(double)));

            for( auto &pi : packInfos_ ){
		// the size as we will compress the data for transfer
	       size_t buffsize = zfp_stream_maximum_size(zfp, zfp_field_1d(NULL, zfp_type_double, pi->size( *dir, block )/sizeof(double)));
               receiverInfo[nProcess] += mpi::MPISize( buffsize );
	    } 
	    zfp_stream_close(zfp);

            auto &headerBuffer = headerExchangeBs.sendBuffer( nProcess );
            nBlockId.toBuffer( headerBuffer );
            headerBuffer << *dir;
         }
      }

      headerExchangeBs.setReceiverInfoFromSendBufferState( false, true );
      headerExchangeBs.sendAll();
      for( auto recvIter = headerExchangeBs.begin(); recvIter != headerExchangeBs.end(); ++recvIter ) {
         auto &headerVector = headers_[recvIter.rank()];
         auto &buffer = recvIter.buffer();
         while ( buffer.size()) {
            Header header;
            header.blockId.fromBuffer( buffer );
            buffer >> header.dir;
            headerVector.push_back( header );
         }
      }

      bufferSystemCPU_.setReceiverInfo( receiverInfo );
      //bufferSystemGPU_.setReceiverInfo( receiverInfo );
      bufferSystemGPU_copy.setReceiverInfo( receiverInfo );

      for( auto it : receiverInfo ) {
         bufferSystemCPU_.sendBuffer( it.first ).resize( size_t(it.second) );
         //bufferSystemGPU_.sendBuffer( it.first ).resize( size_t(it.second) );
	 bufferSystemGPU_copy.sendBuffer( it.first ).resize( size_t(it.second) );
      }

      forestModificationStamp_ = forest->getBlockForest().getModificationStamp();
      setupBeforeNextCommunication_ = false;
   }

   template<typename Stencil>
   void UniformGPUScheme<Stencil>::addPackInfo( const shared_ptr<GeneratedGPUPackInfo> &pi )
   {
      WALBERLA_ASSERT( !communicationInProgress_, "Cannot add pack info while communication is in progress" )
      packInfos_.push_back( pi );
      setupBeforeNextCommunication_ = true;
   }


} // namespace communication
} // namespace cuda
} // namespace walberla
