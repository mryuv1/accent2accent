def training_step(self, batch):
    inputs = batch["content"].unsqueeze(1)
    styles = batch["style"].unsqueeze(1)

    optimizer_g, optimizer_d = self.optimizers()

    # train generator
    # generate images
    self.toggle_optimizer(optimizer_g)
    self.generated_imgs = self(inputs, styles)

    # log sampled images
    sample_imgs = self.generated_imgs[:6]
    grid = torchvision.utils.make_grid(sample_imgs)
    self.logger.experiment.add_image("generated_images", grid, 0)

    # ground truth result (ie: all fake)
    # put on GPU because we created this tensor inside training_loop
    valid = torch.ones(inputs.size(0), 1)
    valid = valid.type_as(inputs)

    # adversarial loss is binary cross-entropy
    g_loss = self.adversarial_loss(self.discriminator(self(inputs, styles)), valid)
    self.log("g_loss", g_loss, prog_bar=True)
    self.manual_backward(g_loss)
    optimizer_g.step()
    optimizer_g.zero_grad()
    self.untoggle_optimizer(optimizer_g)

    # train discriminator
    # Measure discriminator's ability to classify real from generated samples
    self.toggle_optimizer(optimizer_d)

    # how well can it label as real?
    valid = torch.ones(inputs.size(0), 1)
    valid = valid.type_as(inputs)

    real_loss = self.adversarial_loss(self.discriminator(inputs), valid)

    # how well can it label as fake?
    fake = torch.zeros(inputs.size(0), 1)
    fake = fake.type_as(inputs)

    fake_loss = self.adversarial_loss(self.discriminator(self(inputs, styles).detach()), fake)

    # discriminator loss is the average of these
    d_loss = (real_loss + fake_loss) / 2
    self.log("d_loss", d_loss, prog_bar=True)
    self.manual_backward(d_loss)
    optimizer_d.step()
    optimizer_d.zero_grad()
    self.untoggle_optimizer(optimizer_d)