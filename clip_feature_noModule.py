
# x = model.visual.conv1(x)  # shape = [*, width, grid, grid]
# print('conv1',x.shape)
# x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
# x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
# print('x',x.shape)
# x = torch.cat([model.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
# x = x + model.visual.positional_embedding.to(x.dtype)
# x = model.visual.ln_pre(x)

# x = x.permute(1, 0, 2)  # NLD -> LND
# x = model.visual.transformer(x)
# x = x.permute(1, 0, 2)  # LND -> NLD
# print('x',x,x.shape) #[1, 50, 768]
# x = model.visual.ln_post(x[:, :, :])
# print('x',x,x.shape)  #torch.Size([1, 768]) #1, 50, 768
# if model.visual.proj is not None:
#     x = x @ model.visual.proj
# print('x proj',x,x.shape) #[1, 50, 512]

import torch
import clip # TODO: uncomment if you use clip features
import torch.nn as nn


def _apply_clip_text_model(clip_text_model, data, gpu, pin_memory):
    #import clip
    with torch.no_grad():
        try:
            if gpu=='cpu':
                input_x = clip.tokenize(data['raw_text'], truncate=True)#.cuda(gpu, non_blocking=pin_memory)
            else:
                input_x = clip.tokenize(data['raw_text'], truncate=True).cuda(gpu, non_blocking=pin_memory)
        except:
            print('error text',data['raw_text'])
        #print('raw data text',data['raw_text'])
        #print('input_x',input_x)
        x = clip_text_model.token_embedding(input_x).type(
            clip_text_model.dtype)  # [batch_size, n_ctx, d_model]
        x = x + clip_text_model.positional_embedding.type(clip_text_model.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = clip_text_model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = clip_text_model.ln_final(x).type(clip_text_model.dtype)
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        #text = data['raw_text']
        global_x = x[torch.arange(x.shape[0]), input_x.argmax(dim=-1)] @ clip_text_model.text_projection


        x = x @ clip_text_model.text_projection

        #x = x.detach().cpu()
        #input_x = input_x.cpu()

        batch_size, _, dim = x.shape
        prev_n_tokens = data['text'].shape[1]

        input_x = input_x[:, 1:]  # first token is a token of beginning of the sentence
        x = x[:, 1:]  # first token is a token of beginning of the sentence

        new_text = x[:, :prev_n_tokens] #20 is max?
        if gpu=='cpu':
            new_text_mask = torch.zeros(batch_size, prev_n_tokens)#.cuda(gpu, non_blocking=pin_memory)
        else:
            new_text_mask = torch.zeros(batch_size, prev_n_tokens).cuda(gpu, non_blocking=pin_memory)

        for i in range(len(input_x)):
            # take features from the eot embedding (eot_token is the highest number in each sequence)
            n_eot = input_x[i].argmax().item()
            new_text_mask[i, :n_eot] = 1

        # data['text'] = new_text.type(data['text'].dtype)
        # data['mask'] = new_text_mask.type(data['mask'].dtype)
        # print('text feature',data['text'].shape)
    #return data
    return new_text.float(), new_text_mask.bool(), global_x.float()


def _apply_clip_text_model_prompt(clip_text_model, data, gpu, pin_memory):
    #import clip
    with torch.no_grad():
        try:
            if gpu=='cpu':
                input_x = clip.tokenize(data['raw_text'], truncate=True)#.cuda(gpu, non_blocking=pin_memory)
            else:
                input_x = clip.tokenize(data['raw_text'], truncate=True).cuda(gpu, non_blocking=pin_memory)
        except:
            print('error text',data['raw_text'])
        #print('raw data text',data['raw_text'])
        #print('input_x',input_x)
        x = clip_text_model.token_embedding(input_x).type(
            clip_text_model.dtype)  # [batch_size, n_ctx, d_model]
        x = x + clip_text_model.positional_embedding.type(clip_text_model.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = clip_text_model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = clip_text_model.ln_final(x).type(clip_text_model.dtype)
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        #text = data['raw_text']
        global_x = x[torch.arange(x.shape[0]), input_x.argmax(dim=-1)] @ clip_text_model.text_projection
        
        #===== get global end ======

        # local
        batch_size, _, dim = x.shape
        prev_n_tokens = data['text'].shape[1]

        input_x = input_x[:, 1:]  # first token is a token of beginning of the sentence
        x = x[:, 1:]  # first token is a token of beginning of the sentence

        #new_text = x[:, :prev_n_tokens]

        new_text_mask = torch.zeros(batch_size, prev_n_tokens).cuda(gpu, non_blocking=pin_memory)

        # for i in range(len(input_x)):
        #     # take features from the eot embedding (eot_token is the highest number in each sequence)
        #     n_eot = input_x[i].argmax().item()
        #     new_text_mask[i, :n_eot] = 1

        new_text = torch.zeros(batch_size, prev_n_tokens, 512).cuda(gpu, non_blocking=pin_memory)
        
        # for i in range(batch_size):
        #     raw_text = data['raw_text'][i].split()
        #     for j in range(len(raw_text)):
        #         text =  'This is a photo of '+raw_text[j]
        #         #print('text',text)
        #         #raw_text = 'This is a photo of '+text
        #         try:
        #             input_x = clip.tokenize(text, truncate=True).cuda(gpu, non_blocking=pin_memory)
        #         except:
        #             print('error text',text)
        #         #print('raw data text',data['raw_text'])
        #         #print('input_x',input_x)
        #         x = clip_text_model.token_embedding(input_x).type(
        #             clip_text_model.dtype)  # [batch_size, n_ctx, d_model]
        #         x = x + clip_text_model.positional_embedding.type(clip_text_model.dtype)
        #         x = x.permute(1, 0, 2)  # NLD -> LND
        #         x = clip_text_model.transformer(x)
        #         x = x.permute(1, 0, 2)  # LND -> NLD

        #         x = clip_text_model.ln_final(x).type(clip_text_model.dtype)
        #         # x.shape = [batch_size, n_ctx, transformer.width]
        #         # take features from the eot embedding (eot_token is the highest number in each sequence)
        #         #text = data['raw_text']
        #         new_text[i][j] = x[torch.arange(x.shape[0]), input_x.argmax(dim=-1)] @ clip_text_model.text_projection
        #print('raw_text ',data['raw_text'])
        prompt_text = []
        for i in range(batch_size):
            raw_text = data['raw_text'][i].split()
            for j in range(prev_n_tokens):
                try:
                    #text =  'a photo of a '+raw_text[j]
                    text =  raw_text[j]
                    new_text_mask[i][j] = 1
                except:
                    text =  ''
                prompt_text.append(text)
        print('prompt_text',len(prompt_text),prompt_text[0])
        #print('text',text)
        #raw_text = 'This is a photo of '+text
        try:
            input_x = clip.tokenize(prompt_text, truncate=True).cuda(gpu, non_blocking=pin_memory)
        except:
            print('error text',prompt_text)
        #print('raw data text',data['raw_text'])
        print('input_x',input_x.shape)
        # ====== old inference =====
        # x = clip_text_model.token_embedding(input_x).type(
        #     clip_text_model.dtype)  # [batch_size, n_ctx, d_model]
        # x = x + clip_text_model.positional_embedding.type(clip_text_model.dtype)
        # x = x.permute(1, 0, 2)  # NLD -> LND
        # x = clip_text_model.transformer(x)
        # x = x.permute(1, 0, 2)  # LND -> NLD

        # x = clip_text_model.ln_final(x).type(clip_text_model.dtype)
        # # x.shape = [batch_size, n_ctx, transformer.width]
        # # take features from the eot embedding (eot_token is the highest number in each sequence)
        # #text = data['raw_text']
        # new_text = x[torch.arange(x.shape[0]), input_x.argmax(dim=-1)] @ clip_text_model.text_projection
        
        new_text = clip_text_model.encode_text(input_x)
        
        new_text = new_text.view(batch_size,20,512)
        print('raw_txt',data['raw_text'][0])
        #print('new_text',new_text[0])
        print('new_text_mask',new_text_mask[0])
        #new_text_mask = data["mask"].cuda(args.gpu, non_blocking=args.pin_memory)
    #return data

    return new_text.float(), new_text_mask.bool(), global_x.float()


def _apply_clip_image_model(model, image, gpu, pin_memory, resnet):
    with torch.no_grad():
        #image_features =  CLIP_model.visual.forward(image)
        #image_features = image_features.view(batch_size,16,512)
        #conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        width=64
        bn1 = nn.BatchNorm2d(width // 2).cuda(gpu)
        relu1 = nn.ReLU(inplace=True).cuda(gpu)
        #conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        bn2 = nn.BatchNorm2d(width // 2).cuda(gpu)
        relu2 = nn.ReLU(inplace=True).cuda(gpu)
        #conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        bn3 = nn.BatchNorm2d(width).cuda(gpu)
        relu3 = nn.ReLU(inplace=True).cuda(gpu)
        avgpool = nn.AvgPool2d(2).cuda(gpu)
        model.cuda(gpu)
        if resnet:
            x = image
            def stem(x):
                x = relu1(bn1(model.visual.conv1(x)))
                x = relu2(bn2(model.visual.conv2(x)))
                x = relu3(bn3(model.visual.conv3(x)))
                x = avgpool(x)
                return x
            # def stem(x):
            #     x = model.visual.relu1(model.visual.bn1(model.visual.conv1(x)))
            #     x = model.visual.relu2(model.visual.bn2(model.visual.conv2(x)))
            #     x = model.visual.relu3(model.visual.bn3(model.visual.conv3(x)))
            #     x = model.visual.avgpool(x)
            #     return x
            
            x = x.type(model.visual.conv1.weight.dtype)
            #print('x',x.shape)
            x = stem(x)
            x = model.visual.layer1(x)
            x = model.visual.layer2(x)
            x = model.visual.layer3(x)
            x = model.visual.layer4(x) #[1, 4096, 14, 14]

            local_v = x.permute(0, 2, 3, 1)
            #print('x',x,x.shape) #[192, 2048, 7, 7]

            x = model.visual.attnpool(x) #[1, 1024]
            #print('x pooled',x,x.shape) #192, 512
            global_v = x
        else:
            x = image
            x = model.visual.conv1(x)  # shape = [*, width, grid, grid]
            #print('conv1',x.shape)
            x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
            x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
            #print('x',x.shape)
            x = torch.cat([model.visual.class_embedding.to(x.dtype) +\
            torch.zeros(x.shape[0], 1, x.shape[-1], \
            dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
            x = x + model.visual.positional_embedding.to(x.dtype)
            x = model.visual.ln_pre(x)



            x = x.permute(1, 0, 2)  # NLD -> LND
            x = model.visual.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
            #print('x',x,x.shape) #[1, 50, 768]
            #x = model.visual.ln_post(x[:, :, :])
            x = model.visual.ln_post(x)
            #print('x',x,x.shape)  #torch.Size([1, 768]) #1, 50, 768
            if model.visual.proj is not None:
                x = x @ model.visual.proj
            #print('x proj',x,x.shape) #[1, 50, 512]
            x = x.permute(1, 0, 2) #50,32,512
            global_v = x[0]
            local_v = x[1:].permute(1, 0, 2)

    return global_v.float(),local_v.float()